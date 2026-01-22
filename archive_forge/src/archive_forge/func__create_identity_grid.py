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
def _create_identity_grid(size: Tuple[int, int], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    sy, sx = size
    base_grid = torch.empty(1, sy, sx, 2, device=device, dtype=dtype)
    x_grid = torch.linspace((-sx + 1) / sx, (sx - 1) / sx, sx, device=device, dtype=dtype)
    base_grid[..., 0].copy_(x_grid)
    y_grid = torch.linspace((-sy + 1) / sy, (sy - 1) / sy, sy, device=device, dtype=dtype).unsqueeze_(-1)
    base_grid[..., 1].copy_(y_grid)
    return base_grid