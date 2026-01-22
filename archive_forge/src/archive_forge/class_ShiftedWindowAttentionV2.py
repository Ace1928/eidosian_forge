import math
from functools import partial
from typing import Any, Callable, List, Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ..ops.misc import MLP, Permute
from ..ops.stochastic_depth import StochasticDepth
from ..transforms._presets import ImageClassification, InterpolationMode
from ..utils import _log_api_usage_once
from ._api import register_model, Weights, WeightsEnum
from ._meta import _IMAGENET_CATEGORIES
from ._utils import _ovewrite_named_param, handle_legacy_interface
class ShiftedWindowAttentionV2(ShiftedWindowAttention):
    """
    See :func:`shifted_window_attention_v2`.
    """

    def __init__(self, dim: int, window_size: List[int], shift_size: List[int], num_heads: int, qkv_bias: bool=True, proj_bias: bool=True, attention_dropout: float=0.0, dropout: float=0.0):
        super().__init__(dim, window_size, shift_size, num_heads, qkv_bias=qkv_bias, proj_bias=proj_bias, attention_dropout=attention_dropout, dropout=dropout)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True), nn.ReLU(inplace=True), nn.Linear(512, num_heads, bias=False))
        if qkv_bias:
            length = self.qkv.bias.numel() // 3
            self.qkv.bias[length:2 * length].data.zero_()

    def define_relative_position_bias_table(self):
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w], indexing='ij'))
        relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous().unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= self.window_size[0] - 1
        relative_coords_table[:, :, :, 1] /= self.window_size[1] - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(torch.abs(relative_coords_table) + 1.0) / 3.0
        self.register_buffer('relative_coords_table', relative_coords_table)

    def get_relative_position_bias(self) -> torch.Tensor:
        relative_position_bias = _get_relative_position_bias(self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads), self.relative_position_index, self.window_size)
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        return relative_position_bias

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()
        return shifted_window_attention(x, self.qkv.weight, self.proj.weight, relative_position_bias, self.window_size, self.num_heads, shift_size=self.shift_size, attention_dropout=self.attention_dropout, dropout=self.dropout, qkv_bias=self.qkv.bias, proj_bias=self.proj.bias, logit_scale=self.logit_scale, training=self.training)