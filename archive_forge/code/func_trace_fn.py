from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import (
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import (
from torch._functorch.utils import exposed_in, argnums_t
def trace_fn(flat_tangents):
    with fwAD.dual_level():
        flat_duals = tuple((fwAD.make_dual(p, t) for p, t in zip(flat_primals, flat_tangents)))
        duals = tree_unflatten(flat_duals, primals_argspec)
        output = func(*duals)
        tangents = tree_map_only(torch.Tensor, lambda t: fwAD.unpack_dual(t)[1], output)
    return tangents