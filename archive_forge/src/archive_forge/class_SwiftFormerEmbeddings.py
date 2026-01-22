import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
class SwiftFormerEmbeddings(nn.Module):
    """
    Embeddings layer consisting of a single 2D convolutional and batch normalization layer.

    Input: tensor of shape `[batch_size, channels, height, width]`

    Output: tensor of shape `[batch_size, channels, height/stride, width/stride]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int):
        super().__init__()
        patch_size = config.down_patch_size
        stride = config.down_stride
        padding = config.down_pad
        embed_dims = config.embed_dims
        in_chans = embed_dims[index]
        embed_dim = embed_dims[index + 1]
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        stride = stride if isinstance(stride, collections.abc.Iterable) else (stride, stride)
        padding = padding if isinstance(padding, collections.abc.Iterable) else (padding, padding)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(embed_dim, eps=config.batch_norm_eps)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x