import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from ...utils.backbone_utils import load_backbone
from .configuration_vit_hybrid import ViTHybridConfig
class ViTHybridPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        image_size, patch_size = (config.image_size, config.patch_size)
        num_channels, hidden_size = (config.num_channels, config.hidden_size)
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.backbone = load_backbone(config)
        if self.backbone.config.model_type != 'bit':
            raise ValueError(f'Backbone model type {self.backbone.model_type} is not supported.')
        feature_dim = self.backbone.channels[-1]
        if feature_size is None:
            feature_map = config.backbone_featmap_shape
            feature_size = feature_map[-2:]
            feature_dim = feature_map[1]
        else:
            feature_size = feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            feature_dim = self.backbone.channels[-1]
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool=False) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        features = self.backbone(pixel_values).feature_maps[-1]
        embeddings = self.projection(features).flatten(2).transpose(1, 2)
        return embeddings