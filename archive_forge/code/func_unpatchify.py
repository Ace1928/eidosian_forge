import collections.abc
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_vit_mae import ViTMAEConfig
def unpatchify(self, patchified_pixel_values):
    """
        Args:
            patchified_pixel_values (`torch.FloatTensor` of shape `(batch_size, num_patches, patch_size**2 * num_channels)`:
                Patchified pixel values.

        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`:
                Pixel values.
        """
    patch_size, num_channels = (self.config.patch_size, self.config.num_channels)
    num_patches_one_direction = int(patchified_pixel_values.shape[1] ** 0.5)
    if num_patches_one_direction ** 2 != patchified_pixel_values.shape[1]:
        raise ValueError('Make sure that the number of patches can be squared')
    batch_size = patchified_pixel_values.shape[0]
    patchified_pixel_values = patchified_pixel_values.reshape(batch_size, num_patches_one_direction, num_patches_one_direction, patch_size, patch_size, num_channels)
    patchified_pixel_values = torch.einsum('nhwpqc->nchpwq', patchified_pixel_values)
    pixel_values = patchified_pixel_values.reshape(batch_size, num_channels, num_patches_one_direction * patch_size, num_patches_one_direction * patch_size)
    return pixel_values