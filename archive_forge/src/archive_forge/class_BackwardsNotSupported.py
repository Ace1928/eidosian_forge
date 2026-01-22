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
class BackwardsNotSupported(torch.autograd.Function):

    @staticmethod
    def forward(ctx, args_spec, *flat_args):
        args, kwargs = tree_unflatten(flat_args, args_spec)
        return redispatch_prim(args, kwargs)

    @staticmethod
    def backward(ctx, *args):
        raise RuntimeError('backwards not supported on prim')