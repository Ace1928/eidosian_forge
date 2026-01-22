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
def _expand_to_indices_in_relevant_chunk(self, indices, sequence_length):
    start_indices_chunk = (indices[:, -1] // self.chunk_length - self.num_chunks_before) * self.chunk_length
    total_chunk_size = self.chunk_length * (1 + self.num_chunks_before + self.num_chunks_after)
    expanded_start_indices = start_indices_chunk.unsqueeze(-1).expand(indices.shape[0], total_chunk_size)
    chunk_sequence_indices = expanded_start_indices + torch.arange(total_chunk_size, device=indices.device, dtype=torch.long).unsqueeze(0).expand(indices.shape[0], total_chunk_size)
    chunk_sequence_indices = chunk_sequence_indices.flatten() % sequence_length
    indices = indices.unsqueeze(1).expand((indices.shape[0], total_chunk_size, -1)).flatten(0, 1).clone()
    indices[:, -1] = chunk_sequence_indices
    return indices