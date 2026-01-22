from __future__ import annotations
import enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import tv_tensors
from torchvision.transforms.v2._utils import check_type, has_any, is_pure_tensor
from torchvision.utils import _log_api_usage_once
from .functional._utils import _get_kernel
def _needs_transform_list(self, flat_inputs: List[Any]) -> List[bool]:
    needs_transform_list = []
    transform_pure_tensor = not has_any(flat_inputs, tv_tensors.Image, tv_tensors.Video, PIL.Image.Image)
    for inpt in flat_inputs:
        needs_transform = True
        if not check_type(inpt, self._transformed_types):
            needs_transform = False
        elif is_pure_tensor(inpt):
            if transform_pure_tensor:
                transform_pure_tensor = False
            else:
                needs_transform = False
        needs_transform_list.append(needs_transform)
    return needs_transform_list