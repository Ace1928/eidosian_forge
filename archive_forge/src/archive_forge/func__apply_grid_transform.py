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
def _apply_grid_transform(img: torch.Tensor, grid: torch.Tensor, mode: str, fill: _FillTypeJIT) -> torch.Tensor:
    fp = img.dtype == grid.dtype
    float_img = img if fp else img.to(grid.dtype)
    shape = float_img.shape
    if shape[0] > 1:
        grid = grid.expand(shape[0], -1, -1, -1)
    if fill is not None:
        mask = torch.ones((shape[0], 1, shape[2], shape[3]), dtype=float_img.dtype, device=float_img.device)
        float_img = torch.cat((float_img, mask), dim=1)
    float_img = grid_sample(float_img, grid, mode=mode, padding_mode='zeros', align_corners=False)
    if fill is not None:
        float_img, mask = torch.tensor_split(float_img, indices=(-1,), dim=-3)
        mask = mask.expand_as(float_img)
        fill_list = fill if isinstance(fill, (tuple, list)) else [float(fill)]
        fill_img = torch.tensor(fill_list, dtype=float_img.dtype, device=float_img.device).view(1, -1, 1, 1)
        if mode == 'nearest':
            bool_mask = mask < 0.5
            float_img[bool_mask] = fill_img.expand_as(float_img)[bool_mask]
        else:
            float_img = float_img.sub_(fill_img).mul_(mask).add_(fill_img)
    img = float_img.round_().to(img.dtype) if not fp else float_img
    return img