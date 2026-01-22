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
def _maybe_remove_out_wrapper(fn: Callable):
    return inspect.unwrap(fn, stop=lambda f: not hasattr(f, '_torch_decompositions_out_wrapper'))