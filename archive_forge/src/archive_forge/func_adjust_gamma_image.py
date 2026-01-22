from typing import List
import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value
from torchvision.utils import _log_api_usage_once
from ._misc import _num_value_bits, to_dtype_image
from ._type_conversion import pil_to_tensor, to_pil_image
from ._utils import _get_kernel, _register_kernel_internal
@_register_kernel_internal(adjust_gamma, torch.Tensor)
@_register_kernel_internal(adjust_gamma, tv_tensors.Image)
def adjust_gamma_image(image: torch.Tensor, gamma: float, gain: float=1.0) -> torch.Tensor:
    if gamma < 0:
        raise ValueError('Gamma should be a non-negative real number')
    if not torch.is_floating_point(image):
        output = to_dtype_image(image, torch.float32, scale=True).pow_(gamma)
    else:
        output = image.pow(gamma)
    if gain != 1.0:
        output = output.mul_(gain).clamp_(0.0, 1.0)
    return to_dtype_image(output, image.dtype, scale=True)