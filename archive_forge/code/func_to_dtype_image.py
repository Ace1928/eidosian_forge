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
@_register_kernel_internal(to_dtype, torch.Tensor)
@_register_kernel_internal(to_dtype, tv_tensors.Image)
def to_dtype_image(image: torch.Tensor, dtype: torch.dtype=torch.float, scale: bool=False) -> torch.Tensor:
    if image.dtype == dtype:
        return image
    elif not scale:
        return image.to(dtype)
    float_input = image.is_floating_point()
    if torch.jit.is_scripting():
        float_output = torch.tensor(0, dtype=dtype).is_floating_point()
    else:
        float_output = dtype.is_floating_point
    if float_input:
        if float_output:
            return image.to(dtype)
        if image.dtype == torch.float32 and dtype in (torch.int32, torch.int64) or (image.dtype == torch.float64 and dtype == torch.int64):
            raise RuntimeError(f'The conversion from {image.dtype} to {dtype} cannot be performed safely.')
        eps = 0.001
        max_value = float(_max_value(dtype))
        return image.mul(max_value + 1.0 - eps).to(dtype)
    else:
        if float_output:
            return image.to(dtype).mul_(1.0 / _max_value(image.dtype))
        num_value_bits_input = _num_value_bits(image.dtype)
        num_value_bits_output = _num_value_bits(dtype)
        if num_value_bits_input > num_value_bits_output:
            return image.bitwise_right_shift(num_value_bits_input - num_value_bits_output).to(dtype)
        else:
            return image.to(dtype).bitwise_left_shift_(num_value_bits_output - num_value_bits_input)