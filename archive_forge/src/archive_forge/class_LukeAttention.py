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
class LukeAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = LukeSelfAttention(config)
        self.output = LukeSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        raise NotImplementedError('LUKE does not support the pruning of attention heads')

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        word_size = word_hidden_states.size(1)
        self_outputs = self.self(word_hidden_states, entity_hidden_states, attention_mask, head_mask, output_attentions)
        if entity_hidden_states is None:
            concat_self_outputs = self_outputs[0]
            concat_hidden_states = word_hidden_states
        else:
            concat_self_outputs = torch.cat(self_outputs[:2], dim=1)
            concat_hidden_states = torch.cat([word_hidden_states, entity_hidden_states], dim=1)
        attention_output = self.output(concat_self_outputs, concat_hidden_states)
        word_attention_output = attention_output[:, :word_size, :]
        if entity_hidden_states is None:
            entity_attention_output = None
        else:
            entity_attention_output = attention_output[:, word_size:, :]
        outputs = (word_attention_output, entity_attention_output) + self_outputs[2:]
        return outputs