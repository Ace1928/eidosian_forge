from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_mobilevitv2 import MobileViTV2Config
def unfolding(self, feature_map: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    batch_size, in_channels, img_height, img_width = feature_map.shape
    patches = nn.functional.unfold(feature_map, kernel_size=(self.patch_height, self.patch_width), stride=(self.patch_height, self.patch_width))
    patches = patches.reshape(batch_size, in_channels, self.patch_height * self.patch_width, -1)
    return (patches, (img_height, img_width))