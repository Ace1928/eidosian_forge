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
def vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
    lazy_load_decompositions()
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    batch_size, flat_in_dims, flat_args, args_spec = _process_batched_inputs(in_dims, args, func)
    if chunk_size is not None:
        chunks_flat_args = _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size)
        return _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs)
    return _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs)