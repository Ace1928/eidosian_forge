import math
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_layoutlmv2 import LayoutLMv2Config
def _calc_visual_bbox(self, image_feature_pool_shape, bbox, device, final_shape):
    visual_bbox_x = torch.div(torch.arange(0, 1000 * (image_feature_pool_shape[1] + 1), 1000, device=device, dtype=bbox.dtype), self.config.image_feature_pool_shape[1], rounding_mode='floor')
    visual_bbox_y = torch.div(torch.arange(0, 1000 * (self.config.image_feature_pool_shape[0] + 1), 1000, device=device, dtype=bbox.dtype), self.config.image_feature_pool_shape[0], rounding_mode='floor')
    visual_bbox = torch.stack([visual_bbox_x[:-1].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[:-1].repeat(image_feature_pool_shape[1], 1).transpose(0, 1), visual_bbox_x[1:].repeat(image_feature_pool_shape[0], 1), visual_bbox_y[1:].repeat(image_feature_pool_shape[1], 1).transpose(0, 1)], dim=-1).view(-1, bbox.size(-1))
    visual_bbox = visual_bbox.repeat(final_shape[0], 1, 1)
    return visual_bbox