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
@_register_kernel_internal(affine, torch.Tensor)
@_register_kernel_internal(affine, tv_tensors.Image)
def affine_image(image: torch.Tensor, angle: Union[int, float], translate: List[float], scale: float, shear: List[float], interpolation: Union[InterpolationMode, int]=InterpolationMode.NEAREST, fill: _FillTypeJIT=None, center: Optional[List[float]]=None) -> torch.Tensor:
    interpolation = _check_interpolation(interpolation)
    if image.numel() == 0:
        return image
    shape = image.shape
    ndim = image.ndim
    if ndim > 4:
        image = image.reshape((-1,) + shape[-3:])
        needs_unsquash = True
    elif ndim == 3:
        image = image.unsqueeze(0)
        needs_unsquash = True
    else:
        needs_unsquash = False
    height, width = shape[-2:]
    angle, translate, shear, center = _affine_parse_args(angle, translate, scale, shear, interpolation, center)
    center_f = [0.0, 0.0]
    if center is not None:
        center_f = [c - s * 0.5 for c, s in zip(center, [width, height])]
    translate_f = [float(t) for t in translate]
    matrix = _get_inverse_affine_matrix(center_f, angle, translate_f, scale, shear)
    _assert_grid_transform_inputs(image, matrix, interpolation.value, fill, ['nearest', 'bilinear'])
    dtype = image.dtype if torch.is_floating_point(image) else torch.float32
    theta = torch.tensor(matrix, dtype=dtype, device=image.device).reshape(1, 2, 3)
    grid = _affine_grid(theta, w=width, h=height, ow=width, oh=height)
    output = _apply_grid_transform(image, grid, interpolation.value, fill=fill)
    if needs_unsquash:
        output = output.reshape(shape)
    return output