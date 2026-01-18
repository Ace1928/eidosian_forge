import math
from typing import List, Optional
import PIL.Image
import torch
from torch.nn.functional import conv2d, pad as torch_pad
from torchvision import tv_tensors
from torchvision.transforms._functional_tensor import _max_value
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
def to_dtype(inpt: torch.Tensor, dtype: torch.dtype=torch.float, scale: bool=False) -> torch.Tensor:
    """[BETA] See :func:`~torchvision.transforms.v2.ToDtype` for details."""
    if torch.jit.is_scripting():
        return to_dtype_image(inpt, dtype=dtype, scale=scale)
    _log_api_usage_once(to_dtype)
    kernel = _get_kernel(to_dtype, type(inpt))
    return kernel(inpt, dtype=dtype, scale=scale)