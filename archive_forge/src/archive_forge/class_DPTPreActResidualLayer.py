import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig
class DPTPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        use_bias_in_fusion_residual = config.use_bias_in_fusion_residual if config.use_bias_in_fusion_residual is not None else not self.use_batch_norm
        self.activation1 = nn.ReLU()
        self.convolution1 = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=use_bias_in_fusion_residual)
        self.activation2 = nn.ReLU()
        self.convolution2 = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=3, stride=1, padding=1, bias=use_bias_in_fusion_residual)
        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.activation1(hidden_state)
        hidden_state = self.convolution1(hidden_state)
        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)
        hidden_state = self.activation2(hidden_state)
        hidden_state = self.convolution2(hidden_state)
        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)
        return hidden_state + residual