import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mgp_str import MgpstrConfig
class MgpstrEmbeddings(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, config: MgpstrConfig):
        super().__init__()
        image_size = config.image_size if isinstance(config.image_size, collections.abc.Iterable) else (config.image_size, config.image_size)
        patch_size = config.patch_size if isinstance(config.patch_size, collections.abc.Iterable) else (config.patch_size, config.patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_tokens = 2 if config.distilled else 1
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_tokens, config.hidden_size))
        self.pos_drop = nn.Dropout(p=config.drop_rate)

    def forward(self, pixel_values):
        batch_size, channel, height, width = pixel_values.shape
        if height != self.image_size[0] or width != self.image_size[1]:
            raise ValueError(f"Input image size ({height}*{width}) doesn't match model ({self.image_size[0]}*{self.image_size[1]}).")
        patch_embeddings = self.proj(pixel_values)
        patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embedding_output = torch.cat((cls_tokens, patch_embeddings), dim=1)
        embedding_output = embedding_output + self.pos_embed
        embedding_output = self.pos_drop(embedding_output)
        return embedding_output