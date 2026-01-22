import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ...generation.logits_process import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import CausalLMOutputWithPast, MaskedLMOutput
from ...modeling_utils import PreTrainedModel, get_parameter_device
from ...utils import (
from ..auto import AutoModel
from .configuration_bark import (
from .generation_configuration_bark import (
class BarkCausalModel(BarkPreTrainedModel):
    config_class = BarkSubModelConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_embeds_layer = nn.Embedding(config.input_vocab_size, config.hidden_size)
        self.position_embeds_layer = nn.Embedding(config.block_size, config.hidden_size)
        self.drop = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([BarkBlock(config, is_causal=True) for _ in range(config.num_layers)])
        self._use_flash_attention_2 = config._attn_implementation == 'flash_attention_2'
        self.layernorm_final = BarkLayerNorm(config.hidden_size, bias=config.bias)
        self.lm_head = nn.Linear(config.hidden_size, config.output_vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        input_embeds = kwargs.get('input_embeds', None)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if past_key_values is not None:
            seq_len = input_ids.shape[1]
            past_length = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]
            input_embeds = None
        elif input_embeds is not None and kwargs.get('use_cache'):
            seq_len = input_embeds.shape[1]
        else:
            seq_len = input_ids.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask[:, :seq_len]
        if position_ids is not None:
            position_ids = position_ids[:, :seq_len]
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]
        else:
            position_ids = None
        if input_embeds is not None and kwargs.get('use_cache'):
            return {'input_ids': None, 'input_embeds': input_embeds, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask}
        return {'input_ids': input_ids, 'past_key_values': past_key_values, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask}

    @add_start_docstrings_to_model_forward(BARK_CAUSAL_MODEL_INPUTS_DOCSTRING)
    def forward(self, input_ids: Optional[torch.Tensor]=None, past_key_values: Optional[Tuple[torch.FloatTensor]]=None, attention_mask: Optional[torch.Tensor]=None, position_ids: Optional[torch.Tensor]=None, head_mask: Optional[torch.Tensor]=None, labels: Optional[torch.LongTensor]=None, input_embeds: Optional[torch.Tensor]=None, use_cache: Optional[bool]=None, output_attentions: Optional[bool]=None, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[Tuple[torch.Tensor], CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if input_ids is not None and input_embeds is not None:
            raise ValueError('You cannot specify both input_ids and input_embeds at the same time')
        elif input_embeds is not None and past_key_values is None:
            pass
        elif input_ids is not None:
            input_embeds = self.input_embeds_layer(input_ids)
        elif input_embeds is not None:
            pass
        else:
            raise ValueError('You have to specify either input_ids or input_embeds')
        input_shape = input_embeds.size()[:-1]
        batch_size = input_embeds.shape[0]
        seq_length = input_shape[-1]
        device = input_ids.device if input_ids is not None else input_embeds.device
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.layers))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        position_embeds = self.position_embeds_layer(position_ids)
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError('batch_size has to be defined and > 0')
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                attention_mask = attention_mask.view(batch_size, -1)
                attention_mask = _prepare_4d_attention_mask(attention_mask, input_embeds.dtype, tgt_len=1)
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        hidden_states = self.drop(input_embeds + position_embeds)
        output_shape = input_shape + (hidden_states.size(-1),)
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once('`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`...')
                use_cache = False
        present_key_values = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, past_layer_key_values) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(block.__call__, hidden_states, None, attention_mask, head_mask[i], use_cache, output_attentions)
            else:
                outputs = block(hidden_states, past_key_values=past_layer_key_values, attention_mask=attention_mask, head_mask=head_mask[i], use_cache=use_cache, output_attentions=output_attentions)
            hidden_states = outputs[0]
            if use_cache:
                present_key_values = present_key_values + (outputs[1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
        hidden_states = self.layernorm_final(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            raise NotImplementedError('Training is not implemented yet for Bark - ensure you do not pass `labels` to the model.')
        if not return_dict:
            return tuple((v for v in [None, logits, present_key_values, all_hidden_states, all_self_attentions] if v is not None))
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=present_key_values, hidden_states=all_hidden_states, attentions=all_self_attentions)

    @staticmethod
    def _reorder_cache(past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple((tuple((past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)) for layer_past in past_key_values))