import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomGrayscale(_RandomApplyTransform):
    """[BETA] Randomly convert image or videos to grayscale with a probability of p (default 0.1).

    .. v2betastatus:: RandomGrayscale transform

    If the input is a :class:`torch.Tensor`, it is expected to have [..., 3 or 1, H, W] shape,
    where ... means an arbitrary number of leading dimensions

    The output has the same number of channels as the input.

    Args:
        p (float): probability that image should be converted to grayscale.
    """
    _v1_transform_cls = _transforms.RandomGrayscale

    def __init__(self, p: float=0.1) -> None:
        super().__init__(p=p)

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_input_channels, *_ = query_chw(flat_inputs)
        return dict(num_input_channels=num_input_channels)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.rgb_to_grayscale, inpt, num_output_channels=params['num_input_channels'])