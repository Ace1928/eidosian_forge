import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def _process_batched_inputs(in_dims: in_dims_t, args: Tuple, func: Callable) -> Tuple[int, List[Any], List[Any], TreeSpec]:
    if not isinstance(in_dims, int) and (not isinstance(in_dims, tuple)):
        raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): expected `in_dims` to be int or a (potentially nested) tuple matching the structure of inputs, got: {type(in_dims)}.')
    if len(args) == 0:
        raise ValueError(f'vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add inputs, or you are trying to vmap over a function with no inputs. The latter is unsupported.')
    flat_args, args_spec = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): in_dims is not compatible with the structure of `inputs`. in_dims has structure {tree_flatten(in_dims)[1]} but inputs has structure {args_spec}.')
    for i, (arg, in_dim) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for an input but in_dim must be either an integer dimension or None.')
        if isinstance(in_dim, int) and (not isinstance(arg, Tensor)):
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for an input but the input is of type {type(arg)}. We cannot vmap over non-Tensor arguments, please use None as the respective in_dim')
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for some input, but that input is a Tensor of dimensionality {arg.dim()} so expected in_dim to satisfy -{arg.dim()} <= in_dim < {arg.dim()}.')
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()
    return (_validate_and_get_batch_size(flat_in_dims, flat_args), flat_in_dims, flat_args, args_spec)