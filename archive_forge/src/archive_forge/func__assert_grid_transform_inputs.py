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
def _assert_grid_transform_inputs(image: torch.Tensor, matrix: Optional[List[float]], interpolation: str, fill: _FillTypeJIT, supported_interpolation_modes: List[str], coeffs: Optional[List[float]]=None) -> None:
    if matrix is not None:
        if not isinstance(matrix, list):
            raise TypeError('Argument matrix should be a list')
        elif len(matrix) != 6:
            raise ValueError('Argument matrix should have 6 float values')
    if coeffs is not None and len(coeffs) != 8:
        raise ValueError('Argument coeffs should have 8 float values')
    if fill is not None:
        if isinstance(fill, (tuple, list)):
            length = len(fill)
            num_channels = image.shape[-3]
            if length > 1 and length != num_channels:
                raise ValueError(f"The number of elements in 'fill' cannot broadcast to match the number of channels of the image ({length} != {num_channels})")
        elif not isinstance(fill, (int, float)):
            raise ValueError('Argument fill should be either int, float, tuple or list')
    if interpolation not in supported_interpolation_modes:
        raise ValueError(f"Interpolation mode '{interpolation}' is unsupported with Tensor input")