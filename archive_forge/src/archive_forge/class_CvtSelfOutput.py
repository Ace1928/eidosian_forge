import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
class CvtSelfOutput(nn.Module):
    """
    The residual connection is defined in CvtLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, embed_dim, drop_rate):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, hidden_state, input_tensor):
        hidden_state = self.dense(hidden_state)
        hidden_state = self.dropout(hidden_state)
        return hidden_state