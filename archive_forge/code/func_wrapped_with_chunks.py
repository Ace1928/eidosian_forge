from torch._functorch.vmap import (vmap_impl, _check_randomness_arg,
from torch._functorch.utils import exposed_in, argnums_t
import functools
@functools.wraps(func)
def wrapped_with_chunks(*args, **kwargs):
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    _, flat_in_dims, flat_args, args_spec = _process_batched_inputs(in_dims, args, func)
    chunks_flat_args = _get_chunk_flat_args(flat_args, flat_in_dims, chunks)
    return _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs)