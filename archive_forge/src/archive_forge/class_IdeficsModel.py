from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PretrainedConfig
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_idefics import IdeficsConfig
from .perceiver import IdeficsPerceiverResampler
from .vision import IdeficsVisionTransformer
@add_start_docstrings('The bare LLaMA Model outputting raw hidden-states without any specific head on top.', LLAMA_START_DOCSTRING)
class IdeficsModel(IdeficsPreTrainedModel):
    """
    Transformer decoder consisting of `config.num_hidden_layers` layers. Each layer is a [`IdeficsDecoderLayer`]

    Args:
        config: IdeficsConfig
    """

    def __init__(self, config: IdeficsConfig):
        super().__init__(config)
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = IdeficsDecoupledEmbedding(num_embeddings=config.vocab_size, num_additional_embeddings=config.additional_vocab_size, embedding_dim=config.hidden_size, partially_freeze=config.freeze_text_layers, padding_idx=self.padding_idx)
        self.image_size = config.vision_config.image_size
        self.vision_config = config.vision_config
        self.vision_model = IdeficsVisionTransformer(config.vision_config)
        if config.use_resampler:
            perceiver_config = config.perceiver_config
            self.perceiver_resampler = IdeficsPerceiverResampler(config, config.vision_config.embed_dim, perceiver_config.resampler_depth, perceiver_config.resampler_n_heads, perceiver_config.resampler_head_dim, perceiver_config.resampler_n_latents)
        self.layers = nn.ModuleList([IdeficsDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.cross_layer_interval = config.cross_layer_interval
        num_cross_layers = config.num_hidden_layers // self.cross_layer_interval
        self.gated_cross_attn_layers = nn.ModuleList([IdeficsGatedCrossAttentionLayer(config) for _ in range(num_cross_layers)])
        self.gradient_checkpointing = False
        self.norm = IdeficsRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()
        self.freeze_relevant_params(config)

    def freeze_relevant_params(self, config=None):
        if config is None:
            config = self.config
        if config.freeze_text_layers:
            self.freeze_text_layers(config.freeze_text_module_exceptions)
        if config.freeze_vision_layers:
            freeze_model(self.vision_model, module_exceptions=config.freeze_vision_module_exceptions)

    def freeze_text_layers(self, module_exceptions=[]):
        for module in [self.layers, self.norm]:
            freeze_model(module, module_exceptions=module_exceptions)

    def freeze_vision_layers(self, module_exceptions=[]):
        freeze_model(self.vision_model, module_exceptions=module_exceptions)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(self, input_ids: torch.LongTensor=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.LongTensor]=None, past_key_values: Optional[List[torch.FloatTensor]]=None, inputs_embeds: Optional[torch.FloatTensor]=None, pixel_values: Optional[torch.FloatTensor]=None, image_encoder_embeddings: Optional[torch.FloatTensor]=None, perceiver_embeddings: Optional[torch.FloatTensor]=None, image_attention_mask: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, interpolate_pos_encoding: Optional[bool]=False, return_dict: Optional[bool]=None) -> Union[Tuple, IdeficsBaseModelOutputWithPast]:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time')
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError('You have to specify either decoder_input_ids or decoder_inputs_embeds')
        seq_length_with_past = seq_length
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        elif position_ids is None:
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        if (pixel_values, image_encoder_embeddings, perceiver_embeddings).count(None) != 2:
            raise ValueError('Exactly 1 of pixel_values, image_encoder_embeddings or perceiver_embeddings has to be not-None.')
        elif pixel_values is not None:
            pixel_values = pixel_values.to(dtype=self.dtype, device=device)
            batch_size, num_images = pixel_values.shape[:2]
            pixel_values = pixel_values.contiguous().view(batch_size * num_images, *pixel_values.shape[2:])
            image_hidden_states = self.vision_model(pixel_values=pixel_values, interpolate_pos_encoding=interpolate_pos_encoding).last_hidden_state
        elif image_encoder_embeddings is not None:
            batch_size, num_images, image_seq_len, image_hidden_size = image_encoder_embeddings.size()
            image_hidden_states = image_encoder_embeddings.to(dtype=self.dtype, device=device)
            image_hidden_states = image_hidden_states.view(batch_size * num_images, image_seq_len, image_hidden_size)
        if self.config.use_resampler:
            if perceiver_embeddings is None:
                perceiver_embeddings = self.perceiver_resampler(image_hidden_states)
                image_seq_len, image_hidden_size = (perceiver_embeddings.size(1), perceiver_embeddings.size(2))
            else:
                batch_size, num_images, image_seq_len, image_hidden_size = perceiver_embeddings.size()
            image_hidden_states = perceiver_embeddings
        elif perceiver_embeddings is None:
            image_seq_len, image_hidden_size = (image_hidden_states.size(1), image_hidden_states.size(2))
        else:
            raise ValueError('If `perceiver_embeddings` are passed, use_resampler should be True')
        image_hidden_states = image_hidden_states.view(batch_size, num_images * image_seq_len, image_hidden_size)
        text_seq_len = image_attention_mask.size(1)
        image_attention_mask = image_attention_mask.unsqueeze(-1)
        image_attention_mask = image_attention_mask.repeat(1, 1, 1, image_seq_len)
        image_attention_mask = image_attention_mask.view(batch_size, text_seq_len, num_images * image_seq_len)
        if image_hidden_states is not None:
            image_batch_size, image_sequence_length, _ = image_hidden_states.size()
            image_hidden_shape = (image_batch_size, image_sequence_length)
            if image_attention_mask is None:
                image_attention_mask = torch.ones(image_hidden_shape, device=device)
            image_attention_mask = self.invert_attention_mask(image_attention_mask)
        else:
            image_attention_mask = None
        cross_attention_gate = (image_attention_mask == 0.0).any(dim=-1).to(dtype=self.dtype).squeeze(dim=1).to(device)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device)
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length)
        hidden_states = inputs_embeds
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            def vblock(main_block, hidden_states, attention_mask, position_ids, past_key_value, image_hidden_states, image_attention_mask, cross_attention_gate, output_attentions, use_cache, layer_idx, cross_layer_interval, gated_cross_attn_layers):
                if layer_idx % cross_layer_interval == 0:
                    xblock = gated_cross_attn_layers[layer_idx // cross_layer_interval]
                    outputs = xblock(hidden_states, attention_mask=attention_mask, image_hidden_states=image_hidden_states, image_attention_mask=image_attention_mask, cross_attention_gate=cross_attention_gate, output_attentions=output_attentions, use_cache=use_cache, past_key_value=None)
                    hidden_states = outputs[0]
                layer_outputs = main_block(hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache)
                return layer_outputs
            if self.gradient_checkpointing and self.training:
                past_key_value = None
                if use_cache:
                    logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                    use_cache = False
                layer_outputs = self._gradient_checkpointing_func(vblock, decoder_layer, hidden_states, attention_mask, position_ids, past_key_value, image_hidden_states, image_attention_mask, cross_attention_gate, output_attentions, use_cache, idx, self.cross_layer_interval, self.gated_cross_attn_layers)
            else:
                layer_outputs = vblock(decoder_layer, hidden_states, attention_mask=attention_mask, position_ids=position_ids, past_key_value=past_key_value, image_hidden_states=image_hidden_states, image_attention_mask=image_attention_mask, cross_attention_gate=cross_attention_gate, output_attentions=output_attentions, use_cache=use_cache, layer_idx=idx, cross_layer_interval=self.cross_layer_interval, gated_cross_attn_layers=self.gated_cross_attn_layers)
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        next_cache = next_decoder_cache if use_cache else None
        image_hidden_states = image_hidden_states.view(batch_size, num_images, image_seq_len, image_hidden_size)
        if not return_dict:
            return tuple((v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, image_hidden_states] if v is not None))
        return IdeficsBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=next_cache, hidden_states=all_hidden_states, attentions=all_self_attns, image_hidden_states=image_hidden_states)