import math
import sys
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from ...integrations.deepspeed import is_deepspeed_available
from ...modeling_outputs import ModelOutput
from ...utils import (
from .configuration_esm import EsmConfig
from .modeling_esm import ESM_START_DOCSTRING, EsmModel, EsmPreTrainedModel
from .openfold_utils import (
class EsmFoldAngleResnet(nn.Module):
    """
    Implements Algorithm 20, lines 11-14
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.linear_in = EsmFoldLinear(config.sequence_dim, config.resnet_dim)
        self.linear_initial = EsmFoldLinear(config.sequence_dim, config.resnet_dim)
        self.layers = nn.ModuleList()
        for _ in range(config.num_resnet_blocks):
            layer = EsmFoldAngleResnetBlock(config)
            self.layers.append(layer)
        self.linear_out = EsmFoldLinear(config.resnet_dim, config.num_angles * 2)
        self.relu = nn.ReLU()

    def forward(self, s: torch.Tensor, s_initial: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s:
                [*, C_hidden] single embedding
            s_initial:
                [*, C_hidden] single embedding as of the start of the StructureModule
        Returns:
            [*, no_angles, 2] predicted angles
        """
        s_initial = self.relu(s_initial)
        s_initial = self.linear_initial(s_initial)
        s = self.relu(s)
        s = self.linear_in(s)
        s = s + s_initial
        for l in self.layers:
            s = l(s)
        s = self.relu(s)
        s = self.linear_out(s)
        s = s.view(s.shape[:-1] + (-1, 2))
        unnormalized_s = s
        norm_denom = torch.sqrt(torch.clamp(torch.sum(s ** 2, dim=-1, keepdim=True), min=self.config.epsilon))
        s = s / norm_denom
        return (unnormalized_s, s)