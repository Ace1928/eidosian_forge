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
def adjust_sharpness(inpt: torch.Tensor, sharpness_factor: float) -> torch.Tensor:
    """[BETA] See :class:`~torchvision.transforms.RandomAdjustSharpness`"""
    if torch.jit.is_scripting():
        return adjust_sharpness_image(inpt, sharpness_factor=sharpness_factor)
    _log_api_usage_once(adjust_sharpness)
    kernel = _get_kernel(adjust_sharpness, type(inpt))
    return kernel(inpt, sharpness_factor=sharpness_factor)