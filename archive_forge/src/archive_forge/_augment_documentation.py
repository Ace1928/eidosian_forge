import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import _log_api_usage_once
from ._utils import _get_kernel, _register_kernel_internal
[BETA] See :class:`~torchvision.transforms.v2.RandomErase` for details.