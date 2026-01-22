import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import torch
import torch.fx
import torch.nn as nn
from ...ops import MLP, StochasticDepth
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
class PositionalEncoding(nn.Module):

    def __init__(self, embed_size: int, spatial_size: Tuple[int, int], temporal_size: int, rel_pos_embed: bool) -> None:
        super().__init__()
        self.spatial_size = spatial_size
        self.temporal_size = temporal_size
        self.class_token = nn.Parameter(torch.zeros(embed_size))
        self.spatial_pos: Optional[nn.Parameter] = None
        self.temporal_pos: Optional[nn.Parameter] = None
        self.class_pos: Optional[nn.Parameter] = None
        if not rel_pos_embed:
            self.spatial_pos = nn.Parameter(torch.zeros(self.spatial_size[0] * self.spatial_size[1], embed_size))
            self.temporal_pos = nn.Parameter(torch.zeros(self.temporal_size, embed_size))
            self.class_pos = nn.Parameter(torch.zeros(embed_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        class_token = self.class_token.expand(x.size(0), -1).unsqueeze(1)
        x = torch.cat((class_token, x), dim=1)
        if self.spatial_pos is not None and self.temporal_pos is not None and (self.class_pos is not None):
            hw_size, embed_size = self.spatial_pos.shape
            pos_embedding = torch.repeat_interleave(self.temporal_pos, hw_size, dim=0)
            pos_embedding.add_(self.spatial_pos.unsqueeze(0).expand(self.temporal_size, -1, -1).reshape(-1, embed_size))
            pos_embedding = torch.cat((self.class_pos.unsqueeze(0), pos_embedding), dim=0).unsqueeze(0)
            x.add_(pos_embedding)
        return x