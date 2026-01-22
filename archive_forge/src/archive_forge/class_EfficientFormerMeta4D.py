import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerMeta4D(nn.Module):

    def __init__(self, config: EfficientFormerConfig, dim: int, drop_path: float=0.0):
        super().__init__()
        pool_size = config.pool_size if config.pool_size is not None else 3
        self.token_mixer = EfficientFormerPooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * config.mlp_expansion_ratio)
        self.mlp = EfficientFormerConvMlp(config, in_features=dim, hidden_features=mlp_hidden_dim, drop=config.hidden_dropout_prob)
        self.drop_path = EfficientFormerDropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.use_layer_scale = config.use_layer_scale
        if config.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(config.layer_scale_init_value * torch.ones(dim), requires_grad=True)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs = self.token_mixer(hidden_states)
        if self.use_layer_scale:
            layer_output = hidden_states + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * outputs)
            layer_output = layer_output + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(layer_output))
        else:
            layer_output = hidden_states + self.drop_path(outputs)
            layer_output = layer_output + self.drop_path(self.mlp(layer_output))
        return layer_output