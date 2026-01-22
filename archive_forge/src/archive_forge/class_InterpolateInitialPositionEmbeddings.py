import collections.abc
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_yolos import YolosConfig
class InterpolateInitialPositionEmbeddings(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def forward(self, pos_embed, img_size=(800, 1344)) -> torch.Tensor:
        cls_pos_embed = pos_embed[:, 0, :]
        cls_pos_embed = cls_pos_embed[:, None]
        det_pos_embed = pos_embed[:, -self.config.num_detection_tokens:, :]
        patch_pos_embed = pos_embed[:, 1:-self.config.num_detection_tokens, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        batch_size, hidden_size, seq_len = patch_pos_embed.shape
        patch_height, patch_width = (self.config.image_size[0] // self.config.patch_size, self.config.image_size[1] // self.config.patch_size)
        patch_pos_embed = patch_pos_embed.view(batch_size, hidden_size, patch_height, patch_width)
        height, width = img_size
        new_patch_heigth, new_patch_width = (height // self.config.patch_size, width // self.config.patch_size)
        patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_patch_heigth, new_patch_width), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.flatten(2).transpose(1, 2)
        scale_pos_embed = torch.cat((cls_pos_embed, patch_pos_embed, det_pos_embed), dim=1)
        return scale_pos_embed