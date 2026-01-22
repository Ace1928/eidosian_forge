from typing import Any, Dict, Optional, Union
import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import is_pure_tensor
class ToImage(Transform):
    """[BETA] Convert a tensor, ndarray, or PIL Image to :class:`~torchvision.tv_tensors.Image`
    ; this does not scale values.

    .. v2betastatus:: ToImage transform

    This transform does not support torchscript.
    """
    _transformed_types = (is_pure_tensor, PIL.Image.Image, np.ndarray)

    def _transform(self, inpt: Union[torch.Tensor, PIL.Image.Image, np.ndarray], params: Dict[str, Any]) -> tv_tensors.Image:
        return F.to_image(inpt)