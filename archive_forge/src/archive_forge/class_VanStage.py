import math
from collections import OrderedDict
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_outputs import (
from ....modeling_utils import PreTrainedModel
from ....utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_van import VanConfig
class VanStage(nn.Module):
    """
    VanStage, consisting of multiple layers.
    """

    def __init__(self, config: VanConfig, in_channels: int, hidden_size: int, patch_size: int, stride: int, depth: int, mlp_ratio: int=4, drop_path_rate: float=0.0):
        super().__init__()
        self.embeddings = VanOverlappingPatchEmbedder(in_channels, hidden_size, patch_size, stride)
        self.layers = nn.Sequential(*[VanLayer(config, hidden_size, mlp_ratio=mlp_ratio, drop_path_rate=drop_path_rate) for _ in range(depth)])
        self.normalization = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        hidden_state = self.embeddings(hidden_state)
        hidden_state = self.layers(hidden_state)
        batch_size, hidden_size, height, width = hidden_state.shape
        hidden_state = hidden_state.flatten(2).transpose(1, 2)
        hidden_state = self.normalization(hidden_state)
        hidden_state = hidden_state.view(batch_size, height, width, hidden_size).permute(0, 3, 1, 2)
        return hidden_state