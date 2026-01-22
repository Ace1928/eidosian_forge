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
@_register_kernel_internal(adjust_contrast, torch.Tensor)
@_register_kernel_internal(adjust_contrast, tv_tensors.Image)
def adjust_contrast_image(image: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError(f'contrast_factor ({contrast_factor}) is not non-negative.')
    c = image.shape[-3]
    if c not in [1, 3]:
        raise TypeError(f'Input image tensor permitted channel values are 1 or 3, but found {c}')
    fp = image.is_floating_point()
    if c == 3:
        grayscale_image = _rgb_to_grayscale_image(image, num_output_channels=1, preserve_dtype=False)
        if not fp:
            grayscale_image = grayscale_image.floor_()
    else:
        grayscale_image = image if fp else image.to(torch.float32)
    mean = torch.mean(grayscale_image, dim=(-3, -2, -1), keepdim=True)
    return _blend(image, mean, contrast_factor)