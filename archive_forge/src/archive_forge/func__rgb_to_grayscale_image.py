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
def _rgb_to_grayscale_image(image: torch.Tensor, num_output_channels: int=1, preserve_dtype: bool=True) -> torch.Tensor:
    if image.shape[-3] == 1:
        return image.clone()
    r, g, b = image.unbind(dim=-3)
    l_img = r.mul(0.2989).add_(g, alpha=0.587).add_(b, alpha=0.114)
    l_img = l_img.unsqueeze(dim=-3)
    if preserve_dtype:
        l_img = l_img.to(image.dtype)
    if num_output_channels == 3:
        l_img = l_img.expand(image.shape)
    return l_img