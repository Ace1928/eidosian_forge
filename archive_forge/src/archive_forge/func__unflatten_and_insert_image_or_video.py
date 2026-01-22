import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
import PIL.Image
import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms import _functional_tensor as _FT
from torchvision.transforms.v2 import AutoAugmentPolicy, functional as F, InterpolationMode, Transform
from torchvision.transforms.v2.functional._geometry import _check_interpolation
from torchvision.transforms.v2.functional._meta import get_size
from torchvision.transforms.v2.functional._utils import _FillType, _FillTypeJIT
from ._utils import _get_fill, _setup_fill_arg, check_type, is_pure_tensor
def _unflatten_and_insert_image_or_video(self, flat_inputs_with_spec: Tuple[List[Any], TreeSpec, int], image_or_video: ImageOrVideo) -> Any:
    flat_inputs, spec, idx = flat_inputs_with_spec
    flat_inputs[idx] = image_or_video
    return tree_unflatten(flat_inputs, spec)