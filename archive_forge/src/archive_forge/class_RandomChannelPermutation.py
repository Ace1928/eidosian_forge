import collections.abc
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
from torchvision import transforms as _transforms
from torchvision.transforms.v2 import functional as F, Transform
from ._transform import _RandomApplyTransform
from ._utils import query_chw
class RandomChannelPermutation(Transform):
    """[BETA] Randomly permute the channels of an image or video

    .. v2betastatus:: RandomChannelPermutation transform
    """

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        num_channels, *_ = query_chw(flat_inputs)
        return dict(permutation=torch.randperm(num_channels))

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._call_kernel(F.permute_channels, inpt, params['permutation'])