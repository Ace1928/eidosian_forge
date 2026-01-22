import copy
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from ...activations import ACT2FN
from ...file_utils import (
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import meshgrid
from ...utils import is_accelerate_available, is_ninja_available, is_torchvision_available, logging, requires_backends
from ...utils.backbone_utils import load_backbone
from .configuration_deta import DetaConfig
def _reset_parameters(self):
    nn.init.constant_(self.sampling_offsets.weight.data, 0.0)
    default_dtype = torch.get_default_dtype()
    thetas = torch.arange(self.n_heads, dtype=torch.int64).to(default_dtype) * (2.0 * math.pi / self.n_heads)
    grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
    grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
    for i in range(self.n_points):
        grid_init[:, :, i, :] *= i + 1
    with torch.no_grad():
        self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
    nn.init.constant_(self.attention_weights.weight.data, 0.0)
    nn.init.constant_(self.attention_weights.bias.data, 0.0)
    nn.init.xavier_uniform_(self.value_proj.weight.data)
    nn.init.constant_(self.value_proj.bias.data, 0.0)
    nn.init.xavier_uniform_(self.output_proj.weight.data)
    nn.init.constant_(self.output_proj.bias.data, 0.0)