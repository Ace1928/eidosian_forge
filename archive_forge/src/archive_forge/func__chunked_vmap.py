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
def _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs):
    chunks_output = []
    rs = torch.get_rng_state() if randomness == 'same' else None
    for flat_args in chunks_flat_args:
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
        if batch_size == 0:
            continue
        if rs is not None:
            torch.set_rng_state(rs)
        chunks_output.append(_flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs))
    flat_output_chunks, arg_spec = _flatten_chunks_output(chunks_output)
    del chunks_output
    flat_output = _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks)
    return tree_unflatten(flat_output, arg_spec)