import collections
import math
from typing import Iterable, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_pvt import PvtConfig
class PvtPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config: PvtConfig, image_size: Union[int, Iterable[int]], patch_size: Union[int, Iterable[int]], stride: int, num_channels: int, hidden_size: int, cls_token: bool=False):
        super().__init__()
        self.config = config
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1 if cls_token else num_patches, hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size)) if cls_token else None
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=stride, stride=patch_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        num_patches = height * width
        if num_patches == self.config.image_size * self.config.image_size:
            return self.position_embeddings
        embeddings = embeddings.reshape(1, height, width, -1).permute(0, 3, 1, 2)
        interpolated_embeddings = F.interpolate(embeddings, size=(height, width), mode='bilinear')
        interpolated_embeddings = interpolated_embeddings.reshape(1, -1, height * width).permute(0, 2, 1)
        return interpolated_embeddings

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        patch_embed = self.projection(pixel_values)
        *_, height, width = patch_embed.shape
        patch_embed = patch_embed.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(patch_embed)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(batch_size, -1, -1)
            embeddings = torch.cat((cls_token, embeddings), dim=1)
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings[:, 1:], height, width)
            position_embeddings = torch.cat((self.position_embeddings[:, :1], position_embeddings), dim=1)
        else:
            position_embeddings = self.interpolate_pos_encoding(self.position_embeddings, height, width)
        embeddings = self.dropout(embeddings + position_embeddings)
        return (embeddings, height, width)