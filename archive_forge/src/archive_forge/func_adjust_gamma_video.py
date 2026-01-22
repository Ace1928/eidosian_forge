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
@_register_kernel_internal(adjust_gamma, tv_tensors.Video)
def adjust_gamma_video(video: torch.Tensor, gamma: float, gain: float=1) -> torch.Tensor:
    return adjust_gamma_image(video, gamma=gamma, gain=gain)