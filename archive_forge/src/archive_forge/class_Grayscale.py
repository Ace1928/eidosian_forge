import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class Grayscale(Transform):
    """[BETA] Convert images or videos to grayscale.

    .. v2betastatus:: Grayscale transform

    If the input is a :class:`torch.Tensor`, it is expected
    to have [..., 3 or 1, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image
    """
    _v1_transform_cls = _transforms.Grayscale

    def __init__(self, num_output_channels: int=1):
        super().__init__()
        self.num_output_channels = num_output_channels

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.rgb_to_grayscale, inpt, num_output_channels=self.num_output_channels)