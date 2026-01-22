import math
from typing import Dict, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_mobilevit import MobileViTConfig
class MobileViTLayer(nn.Module):
    """
    MobileViT block: https://arxiv.org/abs/2110.02178
    """

    def __init__(self, config: MobileViTConfig, in_channels: int, out_channels: int, stride: int, hidden_size: int, num_stages: int, dilation: int=1) -> None:
        super().__init__()
        self.patch_width = config.patch_size
        self.patch_height = config.patch_size
        if stride == 2:
            self.downsampling_layer = MobileViTInvertedResidual(config, in_channels=in_channels, out_channels=out_channels, stride=stride if dilation == 1 else 1, dilation=dilation // 2 if dilation > 1 else 1)
            in_channels = out_channels
        else:
            self.downsampling_layer = None
        self.conv_kxk = MobileViTConvLayer(config, in_channels=in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size)
        self.conv_1x1 = MobileViTConvLayer(config, in_channels=in_channels, out_channels=hidden_size, kernel_size=1, use_normalization=False, use_activation=False)
        self.transformer = MobileViTTransformer(config, hidden_size=hidden_size, num_stages=num_stages)
        self.layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.conv_projection = MobileViTConvLayer(config, in_channels=hidden_size, out_channels=in_channels, kernel_size=1)
        self.fusion = MobileViTConvLayer(config, in_channels=2 * in_channels, out_channels=in_channels, kernel_size=config.conv_kernel_size)

    def unfolding(self, features: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        patch_width, patch_height = (self.patch_width, self.patch_height)
        patch_area = int(patch_width * patch_height)
        batch_size, channels, orig_height, orig_width = features.shape
        new_height = int(math.ceil(orig_height / patch_height) * patch_height)
        new_width = int(math.ceil(orig_width / patch_width) * patch_width)
        interpolate = False
        if new_width != orig_width or new_height != orig_height:
            features = nn.functional.interpolate(features, size=(new_height, new_width), mode='bilinear', align_corners=False)
            interpolate = True
        num_patch_width = new_width // patch_width
        num_patch_height = new_height // patch_height
        num_patches = num_patch_height * num_patch_width
        patches = features.reshape(batch_size * channels * num_patch_height, patch_height, num_patch_width, patch_width)
        patches = patches.transpose(1, 2)
        patches = patches.reshape(batch_size, channels, num_patches, patch_area)
        patches = patches.transpose(1, 3)
        patches = patches.reshape(batch_size * patch_area, num_patches, -1)
        info_dict = {'orig_size': (orig_height, orig_width), 'batch_size': batch_size, 'channels': channels, 'interpolate': interpolate, 'num_patches': num_patches, 'num_patches_width': num_patch_width, 'num_patches_height': num_patch_height}
        return (patches, info_dict)

    def folding(self, patches: torch.Tensor, info_dict: Dict) -> torch.Tensor:
        patch_width, patch_height = (self.patch_width, self.patch_height)
        patch_area = int(patch_width * patch_height)
        batch_size = info_dict['batch_size']
        channels = info_dict['channels']
        num_patches = info_dict['num_patches']
        num_patch_height = info_dict['num_patches_height']
        num_patch_width = info_dict['num_patches_width']
        features = patches.contiguous().view(batch_size, patch_area, num_patches, -1)
        features = features.transpose(1, 3)
        features = features.reshape(batch_size * channels * num_patch_height, num_patch_width, patch_height, patch_width)
        features = features.transpose(1, 2)
        features = features.reshape(batch_size, channels, num_patch_height * patch_height, num_patch_width * patch_width)
        if info_dict['interpolate']:
            features = nn.functional.interpolate(features, size=info_dict['orig_size'], mode='bilinear', align_corners=False)
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.downsampling_layer:
            features = self.downsampling_layer(features)
        residual = features
        features = self.conv_kxk(features)
        features = self.conv_1x1(features)
        patches, info_dict = self.unfolding(features)
        patches = self.transformer(patches)
        patches = self.layernorm(patches)
        features = self.folding(patches, info_dict)
        features = self.conv_projection(features)
        features = self.fusion(torch.cat((residual, features), dim=1))
        return features