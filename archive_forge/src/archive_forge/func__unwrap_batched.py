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
def _unwrap_batched(batched_outputs: Union[Tensor, Tuple[Tensor, ...]], out_dims: out_dims_t, vmap_level: int, batch_size: int, func: Callable) -> Tuple:
    flat_batched_outputs, output_spec = tree_flatten(batched_outputs)

    def incompatible_error():
        raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): out_dims is not compatible with the structure of `outputs`. out_dims has structure {tree_flatten(out_dims)[1]} but outputs has structure {output_spec}.')
    if isinstance(batched_outputs, torch.Tensor):
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()
    flat_outputs = [_maybe_remove_batch_dim(_get_name(func), batched_output, vmap_level, batch_size, out_dim) for batched_output, out_dim in zip(flat_batched_outputs, flat_out_dims)]
    return tree_unflatten(flat_outputs, output_spec)