import inspect
import warnings
from functools import wraps
from itertools import chain
from typing import Callable, NamedTuple, Optional, overload, Sequence, Tuple
import torch
import torch._prims_common as utils
from torch._prims_common import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_flatten, tree_unflatten
def _resize_output_check(out: TensorLikeType, shape: ShapeType):
    if utils.same_shape(out.shape, shape):
        return False
    if out.numel() != 0:
        msg = f'An output with one or more elements was resized since it had shape {str(out.shape)} which does not match the required output shape {{str(shape)}}. This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0).'
        warnings.warn(msg)
    return True