import collections
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_flava import (
class FlavaImageCodebookResPath(nn.Module):

    def __init__(self, in_size: int, out_size: int, **kwargs):
        super().__init__()
        hid_size = out_size // 4
        path = OrderedDict()
        path['relu_1'] = nn.ReLU()
        path['conv_1'] = nn.Conv2d(in_size, hid_size, kernel_size=3, padding=1)
        path['relu_2'] = nn.ReLU()
        path['conv_2'] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)
        path['relu_3'] = nn.ReLU()
        path['conv_3'] = nn.Conv2d(hid_size, hid_size, kernel_size=3, padding=1)
        path['relu_4'] = nn.ReLU()
        path['conv_4'] = nn.Conv2d(hid_size, out_size, kernel_size=1, padding=0)
        self.path = nn.Sequential(path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.path(x)