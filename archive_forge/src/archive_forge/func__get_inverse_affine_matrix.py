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
def _get_inverse_affine_matrix(center: List[float], angle: float, translate: List[float], scale: float, shear: List[float], inverted: bool=True) -> List[float]:
    rot = math.radians(angle)
    sx = math.radians(shear[0])
    sy = math.radians(shear[1])
    cx, cy = center
    tx, ty = translate
    cos_sy = math.cos(sy)
    tan_sx = math.tan(sx)
    rot_minus_sy = rot - sy
    cx_plus_tx = cx + tx
    cy_plus_ty = cy + ty
    a = math.cos(rot_minus_sy) / cos_sy
    b = -(a * tan_sx + math.sin(rot))
    c = math.sin(rot_minus_sy) / cos_sy
    d = math.cos(rot) - c * tan_sx
    if inverted:
        matrix = [d / scale, -b / scale, 0.0, -c / scale, a / scale, 0.0]
        matrix[2] += cx - matrix[0] * cx_plus_tx - matrix[1] * cy_plus_ty
        matrix[5] += cy - matrix[3] * cx_plus_tx - matrix[4] * cy_plus_ty
    else:
        matrix = [a * scale, b * scale, 0.0, c * scale, d * scale, 0.0]
        matrix[2] += cx_plus_tx - matrix[0] * cx - matrix[1] * cy
        matrix[5] += cy_plus_ty - matrix[3] * cx - matrix[4] * cy
    return matrix