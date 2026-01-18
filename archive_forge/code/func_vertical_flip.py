import math
import numbers
import warnings
from typing import Any, List, Optional, Sequence, Tuple, Union
import PIL.Image
import torch
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _pad_symmetric
from torchvision.transforms.functional import (
from torchvision.utils import _log_api_usage_once
from ._meta import _get_size_image_pil, clamp_bounding_boxes, convert_bounding_box_format
from ._utils import _FillTypeJIT, _get_kernel, _register_five_ten_crop_kernel_internal, _register_kernel_internal
def vertical_flip(inpt: torch.Tensor) -> torch.Tensor:
    """[BETA] See :class:`~torchvision.transforms.v2.RandomVerticalFlip` for details."""
    if torch.jit.is_scripting():
        return vertical_flip_image(inpt)
    _log_api_usage_once(vertical_flip)
    kernel = _get_kernel(vertical_flip, type(inpt))
    return kernel(inpt)