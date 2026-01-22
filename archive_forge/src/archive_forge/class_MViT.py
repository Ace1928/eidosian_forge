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
class MViT(nn.Module):

    def __init__(self, spatial_size: Tuple[int, int], temporal_size: int, block_setting: Sequence[MSBlockConfig], residual_pool: bool, residual_with_cls_embed: bool, rel_pos_embed: bool, proj_after_attn: bool, dropout: float=0.5, attention_dropout: float=0.0, stochastic_depth_prob: float=0.0, num_classes: int=400, block: Optional[Callable[..., nn.Module]]=None, norm_layer: Optional[Callable[..., nn.Module]]=None, patch_embed_kernel: Tuple[int, int, int]=(3, 7, 7), patch_embed_stride: Tuple[int, int, int]=(2, 4, 4), patch_embed_padding: Tuple[int, int, int]=(1, 3, 3)) -> None:
        """
        MViT main class.

        Args:
            spatial_size (tuple of ints): The spacial size of the input as ``(H, W)``.
            temporal_size (int): The temporal size ``T`` of the input.
            block_setting (sequence of MSBlockConfig): The Network structure.
            residual_pool (bool): If True, use MViTv2 pooling residual connection.
            residual_with_cls_embed (bool): If True, the addition on the residual connection will include
                the class embedding.
            rel_pos_embed (bool): If True, use MViTv2's relative positional embeddings.
            proj_after_attn (bool): If True, apply the projection after the attention.
            dropout (float): Dropout rate. Default: 0.0.
            attention_dropout (float): Attention dropout rate. Default: 0.0.
            stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
            num_classes (int): The number of classes.
            block (callable, optional): Module specifying the layer which consists of the attention and mlp.
            norm_layer (callable, optional): Module specifying the normalization layer to use.
            patch_embed_kernel (tuple of ints): The kernel of the convolution that patchifies the input.
            patch_embed_stride (tuple of ints): The stride of the convolution that patchifies the input.
            patch_embed_padding (tuple of ints): The padding of the convolution that patchifies the input.
        """
        super().__init__()
        _log_api_usage_once(self)
        total_stage_blocks = len(block_setting)
        if total_stage_blocks == 0:
            raise ValueError("The configuration parameter can't be empty.")
        if block is None:
            block = MultiscaleBlock
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-06)
        self.conv_proj = nn.Conv3d(in_channels=3, out_channels=block_setting[0].input_channels, kernel_size=patch_embed_kernel, stride=patch_embed_stride, padding=patch_embed_padding)
        input_size = [size // stride for size, stride in zip((temporal_size,) + spatial_size, self.conv_proj.stride)]
        self.pos_encoding = PositionalEncoding(embed_size=block_setting[0].input_channels, spatial_size=(input_size[1], input_size[2]), temporal_size=input_size[0], rel_pos_embed=rel_pos_embed)
        self.blocks = nn.ModuleList()
        for stage_block_id, cnf in enumerate(block_setting):
            sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
            self.blocks.append(block(input_size=input_size, cnf=cnf, residual_pool=residual_pool, residual_with_cls_embed=residual_with_cls_embed, rel_pos_embed=rel_pos_embed, proj_after_attn=proj_after_attn, dropout=attention_dropout, stochastic_depth_prob=sd_prob, norm_layer=norm_layer))
            if len(cnf.stride_q) > 0:
                input_size = [size // stride for size, stride in zip(input_size, cnf.stride_q)]
        self.norm = norm_layer(block_setting[-1].output_channels)
        self.head = nn.Sequential(nn.Dropout(dropout, inplace=True), nn.Linear(block_setting[-1].output_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, PositionalEncoding):
                for weights in m.parameters():
                    nn.init.trunc_normal_(weights, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _unsqueeze(x, 5, 2)[0]
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_encoding(x)
        thw = (self.pos_encoding.temporal_size,) + self.pos_encoding.spatial_size
        for block in self.blocks:
            x, thw = block(x, thw)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x