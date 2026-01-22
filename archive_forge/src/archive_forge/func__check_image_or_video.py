import math
import numbers
import warnings
from typing import Any, Callable, Dict, List, Tuple
import PIL.Image
import torch
from torch.nn.functional import one_hot
from torch.utils._pytree import tree_flatten, tree_unflatten
from torchvision import transforms as _transforms, tv_tensors
from torchvision.transforms.v2 import functional as F
from ._transform import _RandomApplyTransform, Transform
from ._utils import _parse_labels_getter, has_any, is_pure_tensor, query_chw, query_size
def _check_image_or_video(self, inpt: torch.Tensor, *, batch_size: int):
    expected_num_dims = 5 if isinstance(inpt, tv_tensors.Video) else 4
    if inpt.ndim != expected_num_dims:
        raise ValueError(f'Expected a batched input with {expected_num_dims} dims, but got {inpt.ndim} dimensions instead.')
    if inpt.shape[0] != batch_size:
        raise ValueError(f'The batch size of the image or video does not match the batch size of the labels: {inpt.shape[0]} != {batch_size}.')