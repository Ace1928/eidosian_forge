import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
class ReformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dropout = config.hidden_dropout_prob
        self.layers = nn.ModuleList([ReformerLayer(config, i) for i in range(config.num_hidden_layers)])
        self.layer_norm = nn.LayerNorm(2 * config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, past_buckets_states=None, use_cache=False, orig_sequence_length=None, output_hidden_states=False, output_attentions=False):
        all_hidden_states = []
        all_attentions = []
        if past_buckets_states is None:
            past_buckets_states = [(None, None) for i in range(len(self.layers))]
        hidden_states = torch.cat([hidden_states, hidden_states], dim=-1)
        hidden_states = _ReversibleFunction.apply(hidden_states, self.layers, attention_mask, head_mask, num_hashes, all_hidden_states, all_attentions, past_buckets_states, use_cache, orig_sequence_length, output_hidden_states, output_attentions)
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        return ReformerEncoderOutput(hidden_states=hidden_states, all_hidden_states=all_hidden_states, all_attentions=all_attentions, past_buckets_states=past_buckets_states)