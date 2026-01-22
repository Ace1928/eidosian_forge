import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig
@add_start_docstrings('A conditional generation model with a language modeling head. Can be used for sequence generation tasks.', PIX2STRUCT_START_DOCSTRING)
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel):
    config_class = Pix2StructConfig
    main_input_name = 'flattened_patches'
    _tied_weights_keys = ['decoder.lm_head.weight']

    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.encoder = Pix2StructVisionModel(config.vision_config)
        self.decoder = Pix2StructTextModel(config.text_config)
        self.is_vqa = config.is_vqa
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    def resize_token_embeddings(self, new_num_tokens: Optional[int]=None) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)
        self.config.text_config.vocab_size = new_num_tokens
        return model_embeds

    def get_decoder(self):
        return self.decoder

    def get_encoder(self):
        return self.encoder

    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_input_ids: Optional[torch.LongTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, head_mask: Optional[torch.FloatTensor]=None, decoder_head_mask: Optional[torch.FloatTensor]=None, cross_attn_head_mask: Optional[torch.Tensor]=None, encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None, labels: Optional[torch.LongTensor]=None, decoder_inputs_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        """
        Returns:

        Example:

        Inference:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, return_tensors="pt")

        >>> # autoregressive generation
        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print(generated_text)
        A stop sign is on a street corner.

        >>> # conditional generation
        >>> text = "A picture of"
        >>> inputs = processor(text=text, images=image, return_tensors="pt", add_special_tokens=False)

        >>> generated_ids = model.generate(**inputs, max_new_tokens=50)
        >>> generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        >>> print(generated_text)
        A picture of a stop sign with a red stop sign
        ```

        Training:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Pix2StructForConditionalGeneration

        >>> processor = AutoProcessor.from_pretrained("google/pix2struct-base")
        >>> model = Pix2StructForConditionalGeneration.from_pretrained("google/pix2struct-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "A stop sign is on the street corner."

        >>> inputs = processor(images=image, return_tensors="pt")
        >>> labels = processor(text=text, return_tensors="pt").input_ids

        >>> # forward pass
        >>> outputs = model(**inputs, labels=labels)
        >>> loss = outputs.loss
        >>> print(f"{loss.item():.5f}")
        5.94282
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.text_config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if encoder_outputs is None:
            encoder_outputs = self.encoder(flattened_patches=flattened_patches, attention_mask=attention_mask, head_mask=head_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        elif return_dict and (not isinstance(encoder_outputs, BaseModelOutput)):
            encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs[0], hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None, attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None)
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and (decoder_inputs_embeds is None):
            decoder_input_ids = self._shift_right(labels)
            decoder_attention_mask = decoder_attention_mask if decoder_attention_mask is not None else decoder_input_ids.ne(self.config.pad_token_id).float()
            decoder_attention_mask[:, 0] = 1
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, attention_mask=decoder_attention_mask, inputs_embeds=decoder_inputs_embeds, past_key_values=past_key_values, encoder_hidden_states=hidden_states, encoder_attention_mask=attention_mask, head_mask=decoder_head_mask, cross_attn_head_mask=cross_attn_head_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, labels=labels, return_dict=return_dict)
        if not return_dict:
            return decoder_outputs + encoder_outputs
        return Seq2SeqLMOutput(loss=decoder_outputs.loss, logits=decoder_outputs.logits, past_key_values=decoder_outputs.past_key_values, decoder_hidden_states=decoder_outputs.hidden_states, decoder_attentions=decoder_outputs.attentions, cross_attentions=decoder_outputs.cross_attentions, encoder_last_hidden_state=encoder_outputs.last_hidden_state, encoder_hidden_states=encoder_outputs.hidden_states, encoder_attentions=encoder_outputs.attentions)

    def prepare_inputs_for_generation(self, input_ids, flattened_patches: Optional[torch.FloatTensor]=None, attention_mask: Optional[torch.FloatTensor]=None, decoder_attention_mask: Optional[torch.BoolTensor]=None, past_key_values=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, use_cache=None, encoder_outputs=None, **kwargs):
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
        return {'flattened_patches': flattened_patches, 'decoder_input_ids': input_ids, 'past_key_values': past_key_values, 'encoder_outputs': encoder_outputs, 'attention_mask': attention_mask, 'decoder_attention_mask': decoder_attention_mask, 'head_mask': head_mask, 'decoder_head_mask': decoder_head_mask, 'cross_attn_head_mask': cross_attn_head_mask, 'use_cache': use_cache}