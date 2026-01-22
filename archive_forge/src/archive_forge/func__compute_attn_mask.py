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
def _compute_attn_mask(self, query_indices, key_indices, attention_mask, query_key_dots_shape, do_standard_self_attention):
    if attention_mask is not None:
        attention_mask = attention_mask.to(torch.bool)[:, None, :]
        if not do_standard_self_attention:
            attention_mask = self._split_seq_length_dim_to(attention_mask, -1, self.chunk_length, 1)
            attention_mask = self._look_adjacent(attention_mask, self.num_chunks_before, self.num_chunks_after)
        attention_mask = attention_mask.unsqueeze(-2).expand(query_key_dots_shape)
    if self.is_decoder is True:
        causal_mask = torch.ge(query_indices.unsqueeze(-1), key_indices.unsqueeze(-2)).to(query_indices.device)
        if attention_mask is not None:
            attention_mask = causal_mask * attention_mask
        else:
            attention_mask = causal_mask
    return attention_mask