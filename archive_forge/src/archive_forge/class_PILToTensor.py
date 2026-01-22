from typing import Any, Dict, Optional, Union
import numpy as np
import PIL.Image
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F, Transform
from torchvision.transforms.v2._utils import is_pure_tensor
class PILToTensor(Transform):
    """[BETA] Convert a PIL Image to a tensor of the same type - this does not scale values.

    .. v2betastatus:: PILToTensor transform

    This transform does not support torchscript.

    Converts a PIL Image (H x W x C) to a Tensor of shape (C x H x W).
    """
    _transformed_types = (PIL.Image.Image,)

    def _transform(self, inpt: PIL.Image.Image, params: Dict[str, Any]) -> torch.Tensor:
        return F.pil_to_tensor(inpt)