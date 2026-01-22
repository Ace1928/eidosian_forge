import collections.abc
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput, BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_vitdet import VitDetConfig
class VitDetEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) to be consumed by a Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = (config.pretrain_image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = image_size[1] // patch_size[1] * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches
        if config.use_absolute_position_embeddings:
            num_positions = num_patches + 1
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        else:
            self.position_embeddings = None
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def get_absolute_positions(self, abs_pos_embeddings, has_cls_token, height, width):
        """
        Calculate absolute positional embeddings. If needed, resize embeddings and remove cls_token dimension for the
        original embeddings.

        Args:
            abs_pos_embeddings (`torch.Tensor`):
                Absolute positional embeddings with (1, num_position, num_channels).
            has_cls_token (`bool`):
                If true, has 1 embedding in abs_pos_embeddings for cls token.
            height (`int`):
                Height of input image tokens.
            width (`int`):
                Width of input image tokens.

        Returns:
            Absolute positional embeddings after processing with shape (1, height, width, num_channels)
        """
        if has_cls_token:
            abs_pos_embeddings = abs_pos_embeddings[:, 1:]
        num_position = abs_pos_embeddings.shape[1]
        size = int(math.sqrt(num_position))
        if size * size != num_position:
            raise ValueError('Absolute position embeddings must be a square number.')
        if size != height or size != width:
            new_abs_pos_embeddings = nn.functional.interpolate(abs_pos_embeddings.reshape(1, size, size, -1).permute(0, 3, 1, 2), size=(height, width), mode='bicubic', align_corners=False)
            return new_abs_pos_embeddings.permute(0, 2, 3, 1)
        else:
            return abs_pos_embeddings.reshape(1, height, width, -1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        num_channels = pixel_values.shape[1]
        if num_channels != self.num_channels:
            raise ValueError(f'Make sure that the channel dimension of the pixel values match with the one set in the configuration. Expected {self.num_channels} but got {num_channels}.')
        embeddings = self.projection(pixel_values)
        if self.position_embeddings is not None:
            embeddings = embeddings.permute(0, 2, 3, 1)
            embeddings = embeddings + self.get_absolute_positions(self.position_embeddings, True, embeddings.shape[1], embeddings.shape[2])
            embeddings = embeddings.permute(0, 3, 1, 2)
        return embeddings