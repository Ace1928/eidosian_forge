import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
class AlignVisionFinalBlockLayer(nn.Module):
    """
    This corresponds to the final phase of each block in the original implementation.
    """

    def __init__(self, config: AlignVisionConfig, in_dim: int, out_dim: int, stride: int, drop_rate: float, id_skip: bool):
        super().__init__()
        self.apply_dropout = stride == 1 and (not id_skip)
        self.project_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1, padding='same', bias=False)
        self.project_bn = nn.BatchNorm2d(num_features=out_dim, eps=config.batch_norm_eps, momentum=config.batch_norm_momentum)
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, embeddings: torch.FloatTensor, hidden_states: torch.FloatTensor) -> torch.Tensor:
        hidden_states = self.project_conv(hidden_states)
        hidden_states = self.project_bn(hidden_states)
        if self.apply_dropout:
            hidden_states = self.dropout(hidden_states)
            hidden_states = hidden_states + embeddings
        return hidden_states