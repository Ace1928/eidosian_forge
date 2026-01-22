import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN, gelu
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_luke import LukeConfig
class LukeEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LukeLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=None, head_mask=None, output_attentions=False, output_hidden_states=False, return_dict=True):
        all_word_hidden_states = () if output_hidden_states else None
        all_entity_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
                all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)
            layer_head_mask = head_mask[i] if head_mask is not None else None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, word_hidden_states, entity_hidden_states, attention_mask, layer_head_mask, output_attentions)
            else:
                layer_outputs = layer_module(word_hidden_states, entity_hidden_states, attention_mask, layer_head_mask, output_attentions)
            word_hidden_states = layer_outputs[0]
            if entity_hidden_states is not None:
                entity_hidden_states = layer_outputs[1]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)
        if output_hidden_states:
            all_word_hidden_states = all_word_hidden_states + (word_hidden_states,)
            all_entity_hidden_states = all_entity_hidden_states + (entity_hidden_states,)
        if not return_dict:
            return tuple((v for v in [word_hidden_states, all_word_hidden_states, all_self_attentions, entity_hidden_states, all_entity_hidden_states] if v is not None))
        return BaseLukeModelOutput(last_hidden_state=word_hidden_states, hidden_states=all_word_hidden_states, attentions=all_self_attentions, entity_last_hidden_state=entity_hidden_states, entity_hidden_states=all_entity_hidden_states)