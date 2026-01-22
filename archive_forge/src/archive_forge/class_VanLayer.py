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
class VanLayer(nn.Module):
    """
    Van layer composed by normalization layers, large kernel attention (LKA) and a multi layer perceptron (MLP).
    """

    def __init__(self, config: VanConfig, hidden_size: int, mlp_ratio: int=4, drop_path_rate: float=0.5):
        super().__init__()
        self.drop_path = VanDropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        self.pre_normomalization = nn.BatchNorm2d(hidden_size)
        self.attention = VanSpatialAttentionLayer(hidden_size, config.hidden_act)
        self.attention_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)
        self.post_normalization = nn.BatchNorm2d(hidden_size)
        self.mlp = VanMlpLayer(hidden_size, hidden_size * mlp_ratio, hidden_size, config.hidden_act, config.dropout_rate)
        self.mlp_scaling = VanLayerScaling(hidden_size, config.layer_scale_init_value)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        residual = hidden_state
        hidden_state = self.pre_normomalization(hidden_state)
        hidden_state = self.attention(hidden_state)
        hidden_state = self.attention_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        hidden_state = residual + hidden_state
        residual = hidden_state
        hidden_state = self.post_normalization(hidden_state)
        hidden_state = self.mlp(hidden_state)
        hidden_state = self.mlp_scaling(hidden_state)
        hidden_state = self.drop_path(hidden_state)
        hidden_state = residual + hidden_state
        return hidden_state