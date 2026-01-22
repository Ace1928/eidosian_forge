import warnings
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor
from torch.nn.functional import conv2d, grid_sample, interpolate, pad as torch_pad
def _blurred_degenerate_image(img: Tensor) -> Tensor:
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    kernel = torch.ones((3, 3), dtype=dtype, device=img.device)
    kernel[1, 1] = 5.0
    kernel /= kernel.sum()
    kernel = kernel.expand(img.shape[-3], 1, kernel.shape[0], kernel.shape[1])
    result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in(img, [kernel.dtype])
    result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
    result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)
    result = img.clone()
    result[..., 1:-1, 1:-1] = result_tmp
    return result