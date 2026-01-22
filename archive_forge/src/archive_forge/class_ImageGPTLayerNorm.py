import math
import os
import warnings
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.cuda.amp import autocast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import Conv1D, find_pruneable_heads_and_indices, prune_conv1d_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_imagegpt import ImageGPTConfig
class ImageGPTLayerNorm(nn.Module):

    def __init__(self, hidden_size: Tuple[int], eps: float=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(hidden_size))

    def forward(self, tensor: torch.Tensor) -> tuple:
        return tensor / torch.sqrt(torch.mean(torch.square(tensor), axis=-1, keepdim=True) + self.eps) * self.weight.data[..., :]