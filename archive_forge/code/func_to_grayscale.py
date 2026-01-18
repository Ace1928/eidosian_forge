import math
import numbers
import warnings
from enum import Enum
from typing import Any, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from ..utils import _log_api_usage_once
from . import _functional_pil as F_pil, _functional_tensor as F_t
@torch.jit.unused
def to_grayscale(img, num_output_channels=1):
    """Convert PIL image of any mode (RGB, HSV, LAB, etc) to grayscale version of image.
    This transform does not support torch Tensor.

    Args:
        img (PIL Image): PIL Image to be converted to grayscale.
        num_output_channels (int): number of channels of the output image. Value can be 1 or 3. Default is 1.

    Returns:
        PIL Image: Grayscale version of the image.

        - if num_output_channels = 1 : returned image is single channel
        - if num_output_channels = 3 : returned image is 3 channel with r = g = b
    """
    if not torch.jit.is_scripting() and (not torch.jit.is_tracing()):
        _log_api_usage_once(to_grayscale)
    if isinstance(img, Image.Image):
        return F_pil.to_grayscale(img, num_output_channels)
    raise TypeError('Input should be PIL Image')