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
def _flatten_and_extract_image_or_video(self, inputs: Any, unsupported_types: Tuple[Type, ...]=(tv_tensors.BoundingBoxes, tv_tensors.Mask)) -> Tuple[Tuple[List[Any], TreeSpec, int], ImageOrVideo]:
    flat_inputs, spec = tree_flatten(inputs if len(inputs) > 1 else inputs[0])
    needs_transform_list = self._needs_transform_list(flat_inputs)
    image_or_videos = []
    for idx, (inpt, needs_transform) in enumerate(zip(flat_inputs, needs_transform_list)):
        if needs_transform and check_type(inpt, (tv_tensors.Image, PIL.Image.Image, is_pure_tensor, tv_tensors.Video)):
            image_or_videos.append((idx, inpt))
        elif isinstance(inpt, unsupported_types):
            raise TypeError(f'Inputs of type {type(inpt).__name__} are not supported by {type(self).__name__}()')
    if not image_or_videos:
        raise TypeError('Found no image in the sample.')
    if len(image_or_videos) > 1:
        raise TypeError(f'Auto augment transformations are only properly defined for a single image or video, but found {len(image_or_videos)}.')
    idx, image_or_video = image_or_videos[0]
    return ((flat_inputs, spec, idx), image_or_video)