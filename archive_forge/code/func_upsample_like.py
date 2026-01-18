import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
def upsample_like(pixel_values: Tensor, like: Tensor, mode: str='bilinear') -> Tensor:
    """
    An utility function that upsamples `pixel_values` to match the dimension of `like`.

    Args:
        pixel_values (`torch.Tensor`):
            The tensor we wish to upsample.
        like (`torch.Tensor`):
            The tensor we wish to use as size target.
        mode (str, *optional*, defaults to `"bilinear"`):
            The interpolation mode.

    Returns:
        `torch.Tensor`: The upsampled tensor
    """
    _, _, height, width = like.shape
    upsampled = nn.functional.interpolate(pixel_values, size=(height, width), mode=mode, align_corners=False)
    return upsampled