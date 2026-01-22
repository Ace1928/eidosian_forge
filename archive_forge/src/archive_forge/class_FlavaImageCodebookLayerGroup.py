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
class FlavaImageCodebookLayerGroup(nn.Module):

    def __init__(self, num_blocks: int, num_layers: int, in_size: int, out_size: int, use_pool: bool=True):
        super().__init__()
        blocks = OrderedDict()
        for i in range(num_blocks):
            if i == 0:
                blocks[f'block_{i + 1}'] = FlavaImageCodebookBlock(in_size, out_size, num_layers)
            else:
                blocks[f'block_{i + 1}'] = FlavaImageCodebookBlock(out_size, out_size, num_layers)
        if use_pool:
            blocks['pool'] = nn.MaxPool2d(kernel_size=2)
        self.group = nn.Sequential(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group(x)