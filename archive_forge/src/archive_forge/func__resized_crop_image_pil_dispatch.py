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
@_register_kernel_internal(resized_crop, PIL.Image.Image)
def _resized_crop_image_pil_dispatch(image: PIL.Image.Image, top: int, left: int, height: int, width: int, size: List[int], interpolation: Union[InterpolationMode, int]=InterpolationMode.BILINEAR, antialias: Optional[Union[str, bool]]='warn') -> PIL.Image.Image:
    if antialias is False:
        warnings.warn('Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.')
    return _resized_crop_image_pil(image, top=top, left=left, height=height, width=width, size=size, interpolation=interpolation)