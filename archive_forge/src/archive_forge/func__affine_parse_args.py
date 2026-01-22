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
def _affine_parse_args(angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: InterpolationMode=InterpolationMode.NEAREST, center: Optional[List[float]]=None) -> Tuple[float, List[float], List[float], Optional[List[float]]]:
    if not isinstance(angle, (int, float)):
        raise TypeError('Argument angle should be int or float')
    if not isinstance(translate, (list, tuple)):
        raise TypeError('Argument translate should be a sequence')
    if len(translate) != 2:
        raise ValueError('Argument translate should be a sequence of length 2')
    if scale <= 0.0:
        raise ValueError('Argument scale should be positive')
    if not isinstance(shear, (numbers.Number, (list, tuple))):
        raise TypeError('Shear should be either a single value or a sequence of two values')
    if not isinstance(interpolation, InterpolationMode):
        raise TypeError('Argument interpolation should be a InterpolationMode')
    if isinstance(angle, int):
        angle = float(angle)
    if isinstance(translate, tuple):
        translate = list(translate)
    if isinstance(shear, numbers.Number):
        shear = [shear, 0.0]
    if isinstance(shear, tuple):
        shear = list(shear)
    if len(shear) == 1:
        shear = [shear[0], shear[0]]
    if len(shear) != 2:
        raise ValueError(f'Shear should be a sequence containing two values. Got {shear}')
    if center is not None:
        if not isinstance(center, (list, tuple)):
            raise TypeError('Argument center should be a sequence')
        else:
            center = [float(c) for c in center]
    return (angle, translate, shear, center)