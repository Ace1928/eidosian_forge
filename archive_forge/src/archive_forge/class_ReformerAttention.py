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
class ReformerAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn_layers = config.attn_layers
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if len(set(self.attn_layers)) == 1 and self.attn_layers[0] == 'lsh':
            self.self_attention = LSHSelfAttention(config)
        elif len(set(self.attn_layers)) == 1 and self.attn_layers[0] == 'local':
            self.self_attention = LocalSelfAttention(config)
        elif len(set(self.attn_layers)) == 2 and set(self.attn_layers) == {'lsh', 'local'}:
            if self.attn_layers[self.layer_id] == 'lsh':
                self.self_attention = LSHSelfAttention(config)
            else:
                self.self_attention = LocalSelfAttention(config)
        else:
            raise NotImplementedError(f"Only attn layer types 'lsh' and 'local' exist, but got `config.attn_layers`: {self.attn_layers}. Select attn layer types from ['lsh', 'local'] only.")
        self.output = ReformerSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, num_hashes=None, past_buckets_states=None, use_cache=False, orig_sequence_length=None, output_attentions=False, buckets=None):
        hidden_states = self.layer_norm(hidden_states)
        if past_buckets_states is not None:
            past_buckets_states_layer = past_buckets_states[self.layer_id]
        else:
            past_buckets_states_layer = None
        self_attention_outputs = self.self_attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, num_hashes=num_hashes, past_buckets_states=past_buckets_states_layer, use_cache=use_cache, output_attentions=output_attentions, buckets=buckets)
        if hasattr(self_attention_outputs, 'buckets'):
            buckets = self_attention_outputs.buckets
        else:
            buckets = None
        if use_cache:
            if past_buckets_states[self.layer_id][0] is None:
                past_buckets = buckets[:, :, :, :orig_sequence_length] if buckets is not None and orig_sequence_length > 1 else buckets
            else:
                past_buckets = torch.cat([past_buckets_states[self.layer_id][0], buckets], dim=-1)
            if past_buckets_states[self.layer_id][1] is None:
                past_states = hidden_states[:, :orig_sequence_length]
            else:
                past_states = torch.cat([past_buckets_states[self.layer_id][1], hidden_states], dim=1)
            past_buckets_states[self.layer_id] = (past_buckets, past_states)
        attention_output = self.output(self_attention_outputs.hidden_states)
        return AttentionOutput(hidden_states=attention_output, attention_probs=self_attention_outputs.attention_probs, buckets=buckets)