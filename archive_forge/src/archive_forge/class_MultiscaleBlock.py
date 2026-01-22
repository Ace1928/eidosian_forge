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
class MultiscaleBlock(nn.Module):

    def __init__(self, input_size: List[int], cnf: MSBlockConfig, residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float=0.0, stochastic_depth_prob: float=0.0, norm_layer: Callable[..., nn.Module]=nn.LayerNorm) -> None:
        super().__init__()
        self.proj_after_attn = proj_after_attn
        self.pool_skip: Optional[nn.Module] = None
        if _prod(cnf.stride_q) > 1:
            kernel_skip = [s + 1 if s > 1 else s for s in cnf.stride_q]
            padding_skip = [int(k // 2) for k in kernel_skip]
            self.pool_skip = Pool(nn.MaxPool3d(kernel_skip, stride=cnf.stride_q, padding=padding_skip), None)
        attn_dim = cnf.output_channels if proj_after_attn else cnf.input_channels
        self.norm1 = norm_layer(cnf.input_channels)
        self.norm2 = norm_layer(attn_dim)
        self.needs_transposal = isinstance(self.norm1, nn.BatchNorm1d)
        self.attn = MultiscaleAttention(input_size, cnf.input_channels, attn_dim, cnf.num_heads, kernel_q=cnf.kernel_q, kernel_kv=cnf.kernel_kv, stride_q=cnf.stride_q, stride_kv=cnf.stride_kv, rel_pos_embed=rel_pos_embed, residual_pool=residual_pool, residual_with_cls_embed=residual_with_cls_embed, dropout=dropout, norm_layer=norm_layer)
        self.mlp = MLP(attn_dim, [4 * attn_dim, cnf.output_channels], activation_layer=nn.GELU, dropout=dropout, inplace=None)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, 'row')
        self.project: Optional[nn.Module] = None
        if cnf.input_channels != cnf.output_channels:
            self.project = nn.Linear(cnf.input_channels, cnf.output_channels)

    def forward(self, x: torch.Tensor, thw: Tuple[int, int, int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        x_norm1 = self.norm1(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm1(x)
        x_attn, thw_new = self.attn(x_norm1, thw)
        x = x if self.project is None or not self.proj_after_attn else self.project(x_norm1)
        x_skip = x if self.pool_skip is None else self.pool_skip(x, thw)[0]
        x = x_skip + self.stochastic_depth(x_attn)
        x_norm2 = self.norm2(x.transpose(1, 2)).transpose(1, 2) if self.needs_transposal else self.norm2(x)
        x_proj = x if self.project is None or self.proj_after_attn else self.project(x_norm2)
        return (x_proj + self.stochastic_depth(self.mlp(x_norm2)), thw_new)