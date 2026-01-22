import contextlib
import itertools
import operator
import weakref
from enum import Enum
from functools import partial, reduce
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type, Union
import torch
import torch._prims_common as utils
import torch.library
from torch import sym_float, Tensor, TypedStorage
from torch._C import _get_default_device
from torch._prims.debug_prims import register_debug_prims
from torch._prims.rng_prims import register_rng_prims
from torch._prims_common import (
from torch._prims_common.wrappers import backwards_not_supported
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode
from torch.overrides import handle_torch_function, has_torch_function
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten
def _reshape_meta(a: TensorLikeType, shape: ShapeType):
    assert isinstance(a, TensorLike)
    utils.validate_shape(shape)
    numel = reduce(operator.mul, shape)
    if numel != a.numel():
        msg = f'Attempting to reshape a tensor with {a.numel()} elements to a shape with {numel} elements!'
        raise ValueError(msg)
    return TensorMeta(a, shape=shape, strides=utils.make_contiguous_strides_for(shape))