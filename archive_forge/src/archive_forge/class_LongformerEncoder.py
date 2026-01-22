import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_longformer import LongformerConfig
class LongformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, attention_mask=None, head_mask=None, padding_len=0, output_attentions=False, output_hidden_states=False, return_dict=True):
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_global_attentions = () if output_attentions and is_global_attn else None
        if head_mask is not None:
            assert head_mask.size()[0] == len(self.layer), f'The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}.'
        for idx, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, attention_mask, head_mask[idx] if head_mask is not None else None, is_index_masked, is_index_global_attn, is_global_attn, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask=attention_mask, layer_head_mask=head_mask[idx] if head_mask is not None else None, is_index_masked=is_index_masked, is_index_global_attn=is_index_global_attn, is_global_attn=is_global_attn, output_attentions=output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1].transpose(1, 2),)
                if is_global_attn:
                    all_global_attentions = all_global_attentions + (layer_outputs[2].transpose(2, 3),)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        hidden_states = hidden_states[:, :hidden_states.shape[1] - padding_len]
        if output_hidden_states:
            all_hidden_states = tuple([state[:, :state.shape[1] - padding_len] for state in all_hidden_states])
        if output_attentions:
            all_attentions = tuple([state[:, :, :state.shape[2] - padding_len, :] for state in all_attentions])
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None))
        return LongformerBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions, global_attentions=all_global_attentions)