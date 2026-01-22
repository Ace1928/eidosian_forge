import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, ModelOutput, QuestionAnsweringModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_splinter import SplinterConfig
class SplinterFullyConnectedLayer(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_act='gelu'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dense = nn.Linear(self.input_dim, self.output_dim)
        self.act_fn = ACT2FN[hidden_act]
        self.LayerNorm = nn.LayerNorm(self.output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(inputs)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states