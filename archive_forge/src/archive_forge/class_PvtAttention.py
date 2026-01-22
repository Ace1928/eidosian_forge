import collections
import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pvt import PvtConfig
class PvtAttention(nn.Module):

    def __init__(self, config: PvtConfig, hidden_size: int, num_attention_heads: int, sequences_reduction_ratio: float):
        super().__init__()
        self.self = PvtEfficientSelfAttention(config, hidden_size=hidden_size, num_attention_heads=num_attention_heads, sequences_reduction_ratio=sequences_reduction_ratio)
        self.output = PvtSelfOutput(config, hidden_size=hidden_size)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states: torch.Tensor, height: int, width: int, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        self_outputs = self.self(hidden_states, height, width, output_attentions)
        attention_output = self.output(self_outputs[0])
        outputs = (attention_output,) + self_outputs[1:]
        return outputs