from functools import partial
from typing import Any, Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ...transforms._presets import VideoClassification
from ...utils import _log_api_usage_once
from .._api import register_model, Weights, WeightsEnum
from .._meta import _KINETICS400_CATEGORIES
from .._utils import _ovewrite_named_param, handle_legacy_interface
from ..swin_transformer import PatchMerging, SwinTransformerBlock
class SwinTransformer3d(nn.Module):
    """
    Implements 3D Swin Transformer from the `"Video Swin Transformer" <https://arxiv.org/abs/2106.13230>`_ paper.
    Args:
        patch_size (List[int]): Patch size.
        embed_dim (int): Patch embedding dimension.
        depths (List(int)): Depth of each Swin Transformer layer.
        num_heads (List(int)): Number of attention heads in different layers.
        window_size (List[int]): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob (float): Stochastic depth rate. Default: 0.1.
        num_classes (int): Number of classes for classification head. Default: 400.
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        block (nn.Module, optional): SwinTransformer Block. Default: None.
        downsample_layer (nn.Module): Downsample layer (patch merging). Default: PatchMerging.
        patch_embed (nn.Module, optional): Patch Embedding layer. Default: None.
    """

    def __init__(self, patch_size: List[int], embed_dim: int, depths: List[int], num_heads: List[int], window_size: List[int], mlp_ratio: float=4.0, dropout: float=0.0, attention_dropout: float=0.0, stochastic_depth_prob: float=0.1, num_classes: int=400, norm_layer: Optional[Callable[..., nn.Module]]=None, block: Optional[Callable[..., nn.Module]]=None, downsample_layer: Callable[..., nn.Module]=PatchMerging, patch_embed: Optional[Callable[..., nn.Module]]=None) -> None:
        super().__init__()
        _log_api_usage_once(self)
        self.num_classes = num_classes
        if block is None:
            block = partial(SwinTransformerBlock, attn_layer=ShiftedWindowAttention3d)
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-05)
        if patch_embed is None:
            patch_embed = PatchEmbed3d
        self.patch_embed = patch_embed(patch_size=patch_size, embed_dim=embed_dim, norm_layer=norm_layer)
        self.pos_drop = nn.Dropout(p=dropout)
        layers: List[nn.Module] = []
        total_stage_blocks = sum(depths)
        stage_block_id = 0
        for i_stage in range(len(depths)):
            stage: List[nn.Module] = []
            dim = embed_dim * 2 ** i_stage
            for i_layer in range(depths[i_stage]):
                sd_prob = stochastic_depth_prob * float(stage_block_id) / (total_stage_blocks - 1)
                stage.append(block(dim, num_heads[i_stage], window_size=window_size, shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size], mlp_ratio=mlp_ratio, dropout=dropout, attention_dropout=attention_dropout, stochastic_depth_prob=sd_prob, norm_layer=norm_layer, attn_layer=ShiftedWindowAttention3d))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if i_stage < len(depths) - 1:
                layers.append(downsample_layer(dim, norm_layer))
        self.features = nn.Sequential(*layers)
        self.num_features = embed_dim * 2 ** (len(depths) - 1)
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(self.num_features, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x