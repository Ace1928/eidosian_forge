import copy
import inspect
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation.configuration_utils import GenerationConfig
from ...generation.logits_process import ClassifierFreeGuidanceLogitsProcessor, LogitsProcessorList
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ..auto.configuration_auto import AutoConfig
from ..auto.modeling_auto import AutoModel
from .configuration_musicgen import MusicgenConfig, MusicgenDecoderConfig
@add_start_docstrings('The composite MusicGen model with a text encoder, audio encoder and Musicgen decoder, for music generation tasks with one or both of text and audio prompts.', MUSICGEN_START_DOCSTRING)
class MusicgenForConditionalGeneration(PreTrainedModel):
    config_class = MusicgenConfig
    base_model_prefix = 'encoder_decoder'
    main_input_name = 'input_ids'
    supports_gradient_checkpointing = True

    def __init__(self, config: Optional[MusicgenConfig]=None, text_encoder: Optional[PreTrainedModel]=None, audio_encoder: Optional[PreTrainedModel]=None, decoder: Optional[MusicgenForCausalLM]=None):
        if config is None and (text_encoder is None or audio_encoder is None or decoder is None):
            raise ValueError('Either a configuration has to be provided, or all three of text encoder, audio encoder and MusicGen decoder.')
        if config is None:
            config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config)
        elif not isinstance(config, self.config_class):
            raise ValueError(f'Config: {config} has to be of type {self.config_class}')
        if config.decoder.cross_attention_hidden_size is not None:
            if config.decoder.cross_attention_hidden_size != config.text_encoder.hidden_size:
                raise ValueError(f"If `cross_attention_hidden_size` is specified in the MusicGen decoder's configuration, it has to be equal to the text encoder's `hidden_size`. Got {config.decoder.cross_attention_hidden_size} for `config.decoder.cross_attention_hidden_size` and {config.text_encoder.hidden_size} for `config.text_encoder.hidden_size`.")
        super().__init__(config)
        if text_encoder is None:
            from ..auto.modeling_auto import AutoModelForTextEncoding
            text_encoder = AutoModelForTextEncoding.from_config(config.text_encoder)
        if audio_encoder is None:
            from ..auto.modeling_auto import AutoModel
            audio_encoder = AutoModel.from_config(config.audio_encoder)
        if decoder is None:
            decoder = MusicgenForCausalLM(config.decoder)
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.decoder = decoder
        if self.text_encoder.config.to_dict() != self.config.text_encoder.to_dict():
            logger.warning(f'Config of the text_encoder: {self.text_encoder.__class__} is overwritten by shared text_encoder config: {self.config.text_encoder}')
        if self.audio_encoder.config.to_dict() != self.config.audio_encoder.to_dict():
            logger.warning(f'Config of the audio_encoder: {self.audio_encoder.__class__} is overwritten by shared audio_encoder config: {self.config.audio_encoder}')
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(f'Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config: {self.config.decoder}')
        self.text_encoder.config = self.config.text_encoder
        self.audio_encoder.config = self.config.audio_encoder
        self.decoder.config = self.config.decoder
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            self.enc_to_dec_proj = nn.Linear(self.text_encoder.config.hidden_size, self.decoder.config.hidden_size)
        if self.text_encoder.get_output_embeddings() is not None:
            raise ValueError(f'The encoder {self.text_encoder} should not have a LM Head. Please use a model without and LM Head')
        decoder_signature = set(inspect.signature(self.decoder.forward).parameters.keys())
        if 'encoder_hidden_states' not in decoder_signature:
            raise ValueError('The selected decoder is not prepared for the encoder hidden states to be passed. Please see the following discussion on GitHub: https://github.com/huggingface/transformers/issues/23350')
        self.tie_weights()

    def tie_weights(self):
        if self.config.tie_encoder_decoder:
            decoder_base_model_prefix = self.decoder.base_model_prefix
            self._tie_encoder_decoder_weights(self.text_encoder, self.decoder._modules[decoder_base_model_prefix], self.decoder.base_model_prefix)

    def get_audio_encoder(self):
        return self.audio_encoder

    def get_text_encoder(self):
        return self.text_encoder

    def get_encoder(self):
        return self.get_text_encoder()

    def get_decoder(self):
        return self.decoder

    def get_input_embeddings(self):
        return self.text_encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        return self.decoder.set_output_embeddings(new_embeddings)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        ```"""
        if kwargs.get('_fast_init', False):
            logger.warning('Fast initialization is currently not supported for MusicgenForConditionalGeneration. Falling back to slow initialization...')
        kwargs['_fast_init'] = False
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    @classmethod
    def from_sub_models_pretrained(cls, text_encoder_pretrained_model_name_or_path: str=None, audio_encoder_pretrained_model_name_or_path: str=None, decoder_pretrained_model_name_or_path: str=None, *model_args, **kwargs) -> PreTrainedModel:
        """
        Instantiate a text encoder, an audio encoder, and a MusicGen decoder from one, two or three base classes of the
        library from pretrained model checkpoints.


        The model is set in evaluation mode by default using `model.eval()` (Dropout modules are deactivated). To train
        the model, you need to first set it back in training mode with `model.train()`.

        Params:
            text_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the text encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            audio_encoder_pretrained_model_name_or_path (`str`, *optional*):
                Information necessary to initiate the audio encoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            decoder_pretrained_model_name_or_path (`str`, *optional*, defaults to `None`):
                Information necessary to initiate the decoder. Can be either:

                    - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    - A path to a *directory* containing model weights saved using
                      [`~PreTrainedModel.save_pretrained`], e.g., `./my_model_directory/`.

            model_args (remaining positional arguments, *optional*):
                All remaining positional arguments will be passed to the underlying model's `__init__` method.

            kwargs (remaining dictionary of keyword arguments, *optional*):
                Can be used to update the configuration object (after it being loaded) and initiate the model (e.g.,
                `output_attentions=True`).

                - To update the text encoder configuration, use the prefix *text_encoder_* for each configuration
                  parameter.
                - To update the audio encoder configuration, use the prefix *audio_encoder_* for each configuration
                  parameter.
                - To update the decoder configuration, use the prefix *decoder_* for each configuration parameter.
                - To update the parent model configuration, do not use a prefix for each configuration parameter.

                Behaves differently depending on whether a `config` is provided or automatically loaded.

        Example:

        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> # initialize a musicgen model from a t5 text encoder, encodec audio encoder, and musicgen decoder
        >>> model = MusicgenForConditionalGeneration.from_sub_models_pretrained(
        ...     text_encoder_pretrained_model_name_or_path="google-t5/t5-base",
        ...     audio_encoder_pretrained_model_name_or_path="facebook/encodec_24khz",
        ...     decoder_pretrained_model_name_or_path="facebook/musicgen-small",
        ... )
        >>> # saving model after fine-tuning
        >>> model.save_pretrained("./musicgen-ft")
        >>> # load fine-tuned model
        >>> model = MusicgenForConditionalGeneration.from_pretrained("./musicgen-ft")
        ```"""
        kwargs_text_encoder = {argument[len('text_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_audio_encoder = {argument[len('audio_encoder_'):]: value for argument, value in kwargs.items() if argument.startswith('audio_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in kwargs.items() if argument.startswith('decoder_')}
        for key in kwargs_text_encoder.keys():
            del kwargs['text_encoder_' + key]
        for key in kwargs_audio_encoder.keys():
            del kwargs['audio_encoder_' + key]
        for key in kwargs_decoder.keys():
            del kwargs['decoder_' + key]
        text_encoder = kwargs_text_encoder.pop('model', None)
        if text_encoder is None:
            if text_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `text_encoder_model` is not defined as an argument, a `text_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_text_encoder:
                encoder_config, kwargs_text_encoder = AutoConfig.from_pretrained(text_encoder_pretrained_model_name_or_path, **kwargs_text_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {text_encoder_pretrained_model_name_or_path} as a text_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_text_encoder['config'] = encoder_config
            text_encoder = AutoModel.from_pretrained(text_encoder_pretrained_model_name_or_path, *model_args, **kwargs_text_encoder)
        audio_encoder = kwargs_audio_encoder.pop('model', None)
        if audio_encoder is None:
            if audio_encoder_pretrained_model_name_or_path is None:
                raise ValueError('If `audio_encoder_model` is not defined as an argument, an `audio_encoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_audio_encoder:
                encoder_config, kwargs_audio_encoder = AutoConfig.from_pretrained(audio_encoder_pretrained_model_name_or_path, **kwargs_audio_encoder, return_unused_kwargs=True)
                if encoder_config.is_decoder is True or encoder_config.add_cross_attention is True:
                    logger.info(f'Initializing {audio_encoder_pretrained_model_name_or_path} as an audio_encoder model from a decoder model. Cross-attention and casual mask are disabled.')
                    encoder_config.is_decoder = False
                    encoder_config.add_cross_attention = False
                kwargs_audio_encoder['config'] = encoder_config
            audio_encoder = AutoModel.from_pretrained(audio_encoder_pretrained_model_name_or_path, *model_args, **kwargs_audio_encoder)
        decoder = kwargs_decoder.pop('model', None)
        if decoder is None:
            if decoder_pretrained_model_name_or_path is None:
                raise ValueError('If `decoder_model` is not defined as an argument, a `decoder_pretrained_model_name_or_path` has to be defined.')
            if 'config' not in kwargs_decoder:
                decoder_config, kwargs_decoder = AutoConfig.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder, return_unused_kwargs=True)
                if isinstance(decoder_config, MusicgenConfig):
                    decoder_config = decoder_config.decoder
                if decoder_config.is_decoder is False or decoder_config.add_cross_attention is False:
                    logger.info(f"Initializing {decoder_pretrained_model_name_or_path} as a decoder model. Cross attention layers are added to {decoder_pretrained_model_name_or_path} and randomly initialized if {decoder_pretrained_model_name_or_path}'s architecture allows for cross attention layers.")
                    decoder_config.is_decoder = True
                    decoder_config.add_cross_attention = True
                kwargs_decoder['config'] = decoder_config
            if kwargs_decoder['config'].is_decoder is False or kwargs_decoder['config'].add_cross_attention is False:
                logger.warning(f'Decoder model {decoder_pretrained_model_name_or_path} is not initialized as a decoder. In order to initialize {decoder_pretrained_model_name_or_path} as a decoder, make sure that the attributes `is_decoder` and `add_cross_attention` of `decoder_config` passed to `.from_sub_models_pretrained(...)` are set to `True` or do not pass a `decoder_config` to `.from_sub_models_pretrained(...)`')
            decoder = MusicgenForCausalLM.from_pretrained(decoder_pretrained_model_name_or_path, **kwargs_decoder)
        config = MusicgenConfig.from_sub_models_config(text_encoder.config, audio_encoder.config, decoder.config, **kwargs)
        return cls(text_encoder=text_encoder, audio_encoder=audio_encoder, decoder=decoder, config=config)

    @add_start_docstrings_to_model_forward(MUSICGEN_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids: Optional[torch.LongTensor]=None, attention_mask: Optional[torch.BoolTensor]=None, input_values: Optional[torch.FloatTensor]=None, padding_mask: Optional[torch.BoolTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, encoder_outputs: Optional[Tuple[torch.FloatTensor]]=None, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, decoder_inputs_embeds: Optional[torch.FloatTensor]=None, labels: Optional[torch.LongTensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None, **kwargs) -> Union[Tuple, Seq2SeqLMOutput]:
        """
        Returns:

        Examples:
        ```python
        >>> from transformers import AutoProcessor, MusicgenForConditionalGeneration
        >>> import torch

        >>> processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        >>> inputs = processor(
        ...     text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
        ...     padding=True,
        ...     return_tensors="pt",
        ... )

        >>> pad_token_id = model.generation_config.pad_token_id
        >>> decoder_input_ids = (
        ...     torch.ones((inputs.input_ids.shape[0] * model.decoder.num_codebooks, 1), dtype=torch.long)
        ...     * pad_token_id
        ... )

        >>> logits = model(**inputs, decoder_input_ids=decoder_input_ids).logits
        >>> logits.shape  # (bsz * num_codebooks, tgt_len, vocab_size)
        torch.Size([8, 1, 2048])
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_text_encoder = {argument[len('text_encoder_')]: value for argument, value in kwargs.items() if argument.startswith('text_encoder_')}
        kwargs_audio_encoder = {argument[len('audio_encoder_')]: value for argument, value in kwargs.items() if argument.startswith('audio_encoder_')}
        kwargs_decoder = {argument[len('decoder_'):]: value for argument, value in kwargs.items() if argument.startswith('decoder_')}
        if encoder_outputs is None:
            encoder_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs_text_encoder)
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)
        encoder_hidden_states = encoder_outputs[0]
        if self.text_encoder.config.hidden_size != self.decoder.config.hidden_size and self.decoder.config.cross_attention_hidden_size is None:
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)
        if attention_mask is not None:
            encoder_hidden_states = encoder_hidden_states * attention_mask[..., None]
        if labels is not None and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        elif decoder_input_ids is None and decoder_inputs_embeds is None:
            audio_encoder_outputs = self.audio_encoder(input_values=input_values, padding_mask=padding_mask, **kwargs_audio_encoder)
            audio_codes = audio_encoder_outputs.audio_codes
            frames, bsz, codebooks, seq_len = audio_codes.shape
            if frames != 1:
                raise ValueError(f'Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is disabled by setting `chunk_length=None` in the audio encoder.')
            if self.config.decoder.audio_channels == 2 and audio_codes.shape[2] == self.decoder.num_codebooks // 2:
                audio_codes = audio_codes.repeat_interleave(2, dim=2)
            decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=attention_mask, inputs_embeds=decoder_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, use_cache=use_cache, past_key_values=past_key_values, return_dict=return_dict, **kwargs_decoder)
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs
        return Seq2SeqLMOutput(loss=loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, decoder_input_ids, past_key_values=None, attention_mask=None, head_mask=None, decoder_attention_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, decoder_delay_pattern_mask=None, guidance_scale=None, **kwargs):
        if decoder_delay_pattern_mask is None:
            decoder_input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(decoder_input_ids, self.generation_config.pad_token_id, max_length=self.generation_config.max_length)
        decoder_input_ids = self.decoder.apply_delay_pattern_mask(decoder_input_ids, decoder_delay_pattern_mask)
        if guidance_scale is not None and guidance_scale > 1:
            decoder_input_ids = decoder_input_ids.repeat((2, 1))
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.repeat((2, 1))
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = decoder_input_ids.shape[1] - 1
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]
        return {'input_ids': None, 'encoder_outputs': encoder_outputs, 'past_key_values': past_key_values, 'decoder_input_ids': decoder_input_ids, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}

    def _prepare_decoder_input_ids_for_generation(self, batch_size: int, model_input_name: str, model_kwargs: Dict[str, torch.Tensor], decoder_start_token_id: int=None, bos_token_id: int=None, device: torch.device=None) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Prepares `decoder_input_ids` for generation with encoder-decoder models"""
        if model_kwargs is not None and 'decoder_input_ids' in model_kwargs:
            decoder_input_ids = model_kwargs.pop('decoder_input_ids')
        elif 'input_ids' in model_kwargs and model_input_name != 'input_ids':
            decoder_input_ids = model_kwargs.pop('input_ids')
        else:
            decoder_input_ids = None
        decoder_start_token_id = self._get_decoder_start_token_id(decoder_start_token_id, bos_token_id)
        if device is None:
            device = self.device
        decoder_input_ids_start = torch.ones((batch_size * self.decoder.num_codebooks, 1), dtype=torch.long, device=device) * decoder_start_token_id
        if decoder_input_ids is None:
            decoder_input_ids = decoder_input_ids_start
        elif (decoder_input_ids[..., 0] != decoder_start_token_id).all().item():
            decoder_input_ids = torch.cat([decoder_input_ids_start, decoder_input_ids], dim=-1)
            if 'decoder_attention_mask' in model_kwargs:
                decoder_attention_mask = model_kwargs['decoder_attention_mask']
                decoder_attention_mask = torch.cat((torch.ones_like(decoder_attention_mask)[:, :1], decoder_attention_mask), dim=-1)
                model_kwargs['decoder_attention_mask'] = decoder_attention_mask
        return (decoder_input_ids, model_kwargs)

    def _prepare_text_encoder_kwargs_for_generation(self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str]=None, guidance_scale: Optional[float]=None) -> Dict[str, Any]:
        encoder = self.get_text_encoder()
        if hasattr(encoder, '_hf_hook'):
            encoder._hf_hook.io_same_device = True
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature}
        model_input_name = model_input_name if model_input_name is not None else self.text_encoder.main_input_name
        encoder_kwargs['return_dict'] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        last_hidden_state = encoder(**encoder_kwargs).last_hidden_state
        if guidance_scale is not None and guidance_scale > 1:
            last_hidden_state = torch.concatenate([last_hidden_state, torch.zeros_like(last_hidden_state)], dim=0)
            if 'attention_mask' in model_kwargs:
                model_kwargs['attention_mask'] = torch.concatenate([model_kwargs['attention_mask'], torch.zeros_like(model_kwargs['attention_mask'])], dim=0)
        model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=last_hidden_state)
        return model_kwargs

    def _prepare_audio_encoder_kwargs_for_generation(self, input_values, model_kwargs, model_input_name: Optional[str]=None):
        encoder = self.get_audio_encoder()
        if hasattr(encoder, '_hf_hook'):
            encoder._hf_hook.io_same_device = True
        irrelevant_prefix = ['decoder_', 'cross_attn', 'use_cache']
        encoder_kwargs = {argument: value for argument, value in model_kwargs.items() if not any((argument.startswith(p) for p in irrelevant_prefix))}
        encoder_signature = set(inspect.signature(encoder.forward).parameters)
        encoder_accepts_wildcard = 'kwargs' in encoder_signature or 'model_kwargs' in encoder_signature
        if not encoder_accepts_wildcard:
            encoder_kwargs = {argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature}
        model_input_name = model_input_name if model_input_name is not None else self.audio_encoder.main_input_name
        encoder_kwargs['return_dict'] = True
        if self.decoder.config.audio_channels == 1:
            encoder_kwargs[model_input_name] = input_values
            audio_encoder_outputs = encoder.encode(**encoder_kwargs)
            audio_codes = audio_encoder_outputs.audio_codes
            audio_scales = audio_encoder_outputs.audio_scales
            frames, bsz, codebooks, seq_len = audio_codes.shape
        else:
            if input_values.shape[1] != 2:
                raise ValueError(f'Expected stereo audio (2-channels) but example has {input_values.shape[1]} channel.')
            encoder_kwargs[model_input_name] = input_values[:, :1, :]
            audio_encoder_outputs_left = encoder.encode(**encoder_kwargs)
            audio_codes_left = audio_encoder_outputs_left.audio_codes
            audio_scales_left = audio_encoder_outputs_left.audio_scales
            encoder_kwargs[model_input_name] = input_values[:, 1:, :]
            audio_encoder_outputs_right = encoder.encode(**encoder_kwargs)
            audio_codes_right = audio_encoder_outputs_right.audio_codes
            audio_scales_right = audio_encoder_outputs_right.audio_scales
            frames, bsz, codebooks, seq_len = audio_codes_left.shape
            audio_codes = audio_codes_left.new_ones((frames, bsz, 2 * codebooks, seq_len))
            audio_codes[:, :, ::2, :] = audio_codes_left
            audio_codes[:, :, 1::2, :] = audio_codes_right
            if audio_scales_left != [None] or audio_scales_right != [None]:
                audio_scales = torch.stack([audio_scales_left, audio_scales_right], dim=1)
            else:
                audio_scales = [None] * bsz
        if frames != 1:
            raise ValueError(f'Expected 1 frame in the audio code outputs, got {frames} frames. Ensure chunking is disabled by setting `chunk_length=None` in the audio encoder.')
        decoder_input_ids = audio_codes[0, ...].reshape(bsz * self.decoder.num_codebooks, seq_len)
        model_kwargs['decoder_input_ids'] = decoder_input_ids
        model_kwargs['audio_scales'] = audio_scales
        return model_kwargs

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    def resize_token_embeddings(self, *args, **kwargs):
        raise NotImplementedError('Resizing the embedding layers via the EncoderDecoderModel directly is not supported. Please use the respective methods of the wrapped objects (model.encoder.resize_token_embeddings(...) or model.decoder.resize_token_embeddings(...))')

    def _maybe_initialize_input_ids_for_generation(self, inputs: Optional[torch.Tensor]=None, bos_token_id: Optional[int]=None, model_kwargs: Optional[Dict[str, torch.Tensor]]=None) -> torch.LongTensor:
        """Initializes input ids for generation, if necessary."""
        if inputs is not None:
            return inputs
        encoder_outputs = model_kwargs.get('encoder_outputs')
        if encoder_outputs is not None:
            shape = encoder_outputs[0].size()[:-1]
            return torch.ones(shape, dtype=torch.long, device=self.device) * -100
        if bos_token_id is None:
            raise ValueError('`bos_token_id` has to be defined when no `input_ids` are provided.')
        batch_size = 1
        for value in model_kwargs.values():
            if isinstance(value, torch.Tensor):
                batch_size = value.shape[0]
                break
        return torch.ones((batch_size, 1), dtype=torch.long, device=self.device) * bos_token_id

    @torch.no_grad()
    def generate(self, inputs: Optional[torch.Tensor]=None, generation_config: Optional[GenerationConfig]=None, logits_processor: Optional[LogitsProcessorList]=None, stopping_criteria: Optional[StoppingCriteriaList]=None, synced_gpus: Optional[bool]=None, streamer: Optional['BaseStreamer']=None, **kwargs):
        """

        Generates sequences of token ids for models with a language modeling head.

        <Tip warning={true}>

        Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
        model's default generation configuration. You can override any `generation_config` by passing the corresponding
        parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

        For an overview of generation strategies and code examples, check out the [following
        guide](./generation_strategies).

        </Tip>

        Parameters:
            inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
                The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
                method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
                should be in the format `input_ids`. For encoder-decoder models *inputs* can represent any of
                `input_ids`, `input_values`, `input_features`, or `pixel_values`.
            generation_config (`~generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, the default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                Custom logits processors that complement the default logits processors built from arguments and
                generation config. If a logit processor is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config. If a stopping criteria is passed that is already created with the arguments or a
                generation config an error is thrown. This feature is intended for advanced users.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
                forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
                specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

        Return:
            [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
            or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

                If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateDecoderOnlyOutput`],
                    - [`~generation.GenerateBeamDecoderOnlyOutput`]

                If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
                [`~utils.ModelOutput`] types are:

                    - [`~generation.GenerateEncoderDecoderOutput`],
                    - [`~generation.GenerateBeamEncoderDecoderOutput`]
        """
        if generation_config is None:
            generation_config = self.generation_config
        generation_config = copy.deepcopy(generation_config)
        model_kwargs = generation_config.update(**kwargs)
        generation_config.validate()
        self._validate_model_kwargs(model_kwargs.copy())
        if model_kwargs.get('encoder_outputs') is not None and type(model_kwargs['encoder_outputs']) == tuple:
            model_kwargs['encoder_outputs'] = BaseModelOutput(last_hidden_state=model_kwargs['encoder_outputs'][0])
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
            if model_kwargs.get('attention_mask', None) is None:
                logger.warning("The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.")
            eos_token_id = generation_config.eos_token_id
            if isinstance(eos_token_id, list):
                eos_token_id = eos_token_id[0]
            logger.warning(f'Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.')
            generation_config.pad_token_id = eos_token_id
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        model_kwargs['output_attentions'] = generation_config.output_attentions
        model_kwargs['output_hidden_states'] = generation_config.output_hidden_states
        model_kwargs['use_cache'] = generation_config.use_cache
        model_kwargs['guidance_scale'] = generation_config.guidance_scale
        requires_attention_mask = 'encoder_outputs' not in model_kwargs
        if model_kwargs.get('attention_mask', None) is None and requires_attention_mask:
            model_kwargs['attention_mask'] = self._prepare_attention_mask_for_generation(inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id)
        if 'encoder_outputs' not in model_kwargs:
            model_kwargs = self._prepare_text_encoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name, guidance_scale=generation_config.guidance_scale)
        if 'decoder_input_ids' not in model_kwargs and 'input_values' in model_kwargs:
            model_kwargs = self._prepare_audio_encoder_kwargs_for_generation(model_kwargs['input_values'], model_kwargs)
        input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(batch_size=batch_size, model_input_name=model_input_name, model_kwargs=model_kwargs, decoder_start_token_id=generation_config.decoder_start_token_id, bos_token_id=generation_config.bos_token_id, device=inputs_tensor.device)
        input_ids_seq_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get('max_length') is None and generation_config.max_length is not None
        if has_default_max_length and generation_config.max_new_tokens is None:
            logger.warning(f'Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.')
        elif generation_config.max_new_tokens is not None:
            if not has_default_max_length:
                logger.warning(f'Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(={generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. Please refer to the documentation for more information. (https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)')
            generation_config.max_length = generation_config.max_new_tokens + input_ids_seq_length
        if generation_config.min_length is not None and generation_config.min_length > generation_config.max_length:
            raise ValueError(f'Unfeasible length constraints: the minimum length ({generation_config.min_length}) is larger than the maximum length ({generation_config.max_length})')
        if input_ids_seq_length >= generation_config.max_length:
            logger.warning(f'Input length of decoder_input_ids is {input_ids_seq_length}, but `max_length` is set to {generation_config.max_length}. This can lead to unexpected behavior. You should consider increasing `max_new_tokens`.')
        input_ids, decoder_delay_pattern_mask = self.decoder.build_delay_pattern_mask(input_ids, pad_token_id=generation_config.decoder_start_token_id, max_length=generation_config.max_length)
        model_kwargs['decoder_delay_pattern_mask'] = decoder_delay_pattern_mask
        if streamer is not None:
            streamer.put(input_ids.cpu())
        is_greedy_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is False)
        is_sample_gen_mode = generation_config.num_beams == 1 and generation_config.num_beam_groups == 1 and (generation_config.do_sample is True)
        if generation_config.guidance_scale is not None and generation_config.guidance_scale > 1:
            logits_processor.append(ClassifierFreeGuidanceLogitsProcessor(generation_config.guidance_scale))
            generation_config.guidance_scale = None
        logits_processor = self._get_logits_processor(generation_config=generation_config, input_ids_seq_length=input_ids_seq_length, encoder_input_ids=inputs_tensor, prefix_allowed_tokens_fn=None, logits_processor=logits_processor)
        stopping_criteria = self._get_stopping_criteria(generation_config=generation_config, stopping_criteria=stopping_criteria)
        if is_greedy_gen_mode:
            if generation_config.num_return_sequences > 1:
                raise ValueError(f'num_return_sequences has to be 1 when doing greedy search, but is {generation_config.num_return_sequences}.')
            outputs = self.greedy_search(input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        elif is_sample_gen_mode:
            logits_warper = self._get_logits_warper(generation_config)
            input_ids, model_kwargs = self._expand_inputs_for_generation(input_ids=input_ids, expand_size=generation_config.num_return_sequences, is_encoder_decoder=self.config.is_encoder_decoder, **model_kwargs)
            outputs = self.sample(input_ids, logits_processor=logits_processor, logits_warper=logits_warper, stopping_criteria=stopping_criteria, pad_token_id=generation_config.pad_token_id, eos_token_id=generation_config.eos_token_id, output_scores=generation_config.output_scores, return_dict_in_generate=generation_config.return_dict_in_generate, synced_gpus=synced_gpus, streamer=streamer, **model_kwargs)
        else:
            raise ValueError('Got incompatible mode for generation, should be one of greedy or sampling. Ensure that beam search is de-activated by setting `num_beams=1` and `num_beam_groups=1`.')
        if generation_config.return_dict_in_generate:
            output_ids = outputs.sequences
        else:
            output_ids = outputs
        output_ids = self.decoder.apply_delay_pattern_mask(output_ids, model_kwargs['decoder_delay_pattern_mask'])
        output_ids = output_ids[output_ids != generation_config.pad_token_id].reshape(batch_size, self.decoder.num_codebooks, -1)
        output_ids = output_ids[None, ...]
        audio_scales = model_kwargs.get('audio_scales')
        if audio_scales is None:
            audio_scales = [None] * batch_size
        if self.decoder.config.audio_channels == 1:
            output_values = self.audio_encoder.decode(output_ids, audio_scales=audio_scales).audio_values
        else:
            codec_outputs_left = self.audio_encoder.decode(output_ids[:, :, ::2, :], audio_scales=audio_scales)
            output_values_left = codec_outputs_left.audio_values
            codec_outputs_right = self.audio_encoder.decode(output_ids[:, :, 1::2, :], audio_scales=audio_scales)
            output_values_right = codec_outputs_right.audio_values
            output_values = torch.cat([output_values_left, output_values_right], dim=1)
        if generation_config.return_dict_in_generate:
            outputs.sequences = output_values
            return outputs
        else:
            return output_values

    def get_unconditional_inputs(self, num_samples=1):
        """
        Helper function to get null inputs for unconditional generation, enabling the model to be used without the
        feature extractor or tokenizer.

        Args:
            num_samples (int, *optional*):
                Number of audio samples to unconditionally generate.
            max_new_tokens (int, *optional*):
                Number of tokens to generate for each sample. More tokens means longer audio samples, at the expense of
                longer inference (since more audio tokens need to be generated per sample).

        Example:
        ```python
        >>> from transformers import MusicgenForConditionalGeneration

        >>> model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        >>> # get the unconditional (or 'null') inputs for the model
        >>> unconditional_inputs = model.get_unconditional_inputs(num_samples=1)
        >>> audio_samples = model.generate(**unconditional_inputs, max_new_tokens=256)
        ```"""
        last_hidden_state = torch.zeros((num_samples, 1, self.config.text_encoder.hidden_size), device=self.device, dtype=self.dtype)
        attention_mask = torch.zeros((num_samples, 1), device=self.device, dtype=torch.long)
        return MusicgenUnconditionalInput(encoder_outputs=(last_hidden_state,), attention_mask=attention_mask, guidance_scale=1.0)