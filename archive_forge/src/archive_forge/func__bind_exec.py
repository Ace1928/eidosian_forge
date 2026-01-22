import logging
import numpy as np
from .base import mx_real_t
from . import ndarray as nd
from .context import cpu
from .io import DataDesc
def _bind_exec(sym, ctx, input_shapes, param_names, need_grad=False, base_exec=None, shared_data_arrays=None, input_types=None, logger=logging):
    """bind executor for bucketing, potentially sharing data with an existing executor."""
    arg_shape, _, aux_shape = sym.infer_shape(**input_shapes)
    assert arg_shape is not None
    if input_types is None:
        input_types = {k: mx_real_t for k in input_shapes.keys()}
    arg_types, _, aux_types = sym.infer_type(**input_types)
    assert arg_types is not None
    arg_arrays = []
    grad_arrays = {} if need_grad is not False else None
    arg_names = sym.list_arguments()
    if need_grad is False:
        need_grad = set()
    elif need_grad is True:
        need_grad = set(arg_names) - set(input_shapes.keys())
    elif isinstance(need_grad, set):
        pass
    else:
        raise AssertionError('need_grad must be boolean or set.')
    grad_req = {name: 'write' if name in need_grad else 'null' for name in arg_names}
    for i, name in enumerate(arg_names):
        if not name in param_names:
            if shared_data_arrays is not None and name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]
                if np.prod(arg_arr.shape) >= np.prod(arg_shape[i]):
                    assert arg_types[i] == arg_arr.dtype
                    arg_arr = arg_arr.reshape(arg_shape[i])
                else:
                    logger.warning('bucketing: data "%s" has a shape %s' % (name, arg_shape[i]) + ', which is larger than already allocated ' + 'shape %s' % (arg_arr.shape,) + '. Need to re-allocate. Consider putting ' + 'default_bucket_key to be the bucket taking the largest ' + 'input for better memory sharing.')
                    arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if shared_data_arrays is not None:
                    shared_data_arrays[name] = arg_arr
            arg_arrays.append(arg_arr)
        else:
            if base_exec is None:
                arg_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                if name in need_grad:
                    grad_arr = nd.zeros(arg_shape[i], ctx, dtype=arg_types[i])
                    grad_arrays[name] = grad_arr
            else:
                arg_arr = base_exec.arg_dict[name]
                assert arg_arr.shape == arg_shape[i]
                assert arg_arr.dtype == arg_types[i]
                if name in need_grad:
                    grad_arrays[name] = base_exec.grad_dict[name]
            arg_arrays.append(arg_arr)
    if base_exec is None:
        aux_arrays = [nd.zeros(s, ctx, dtype=t) for s, t in zip(aux_shape, aux_types)]
    else:
        for i, a in enumerate(base_exec.aux_arrays):
            assert aux_shape[i] == a.shape
            assert aux_types[i] == a.dtype
        aux_arrays = [a for a in base_exec.aux_arrays]
    executor = sym.bind(ctx=ctx, args=arg_arrays, args_grad=grad_arrays, aux_states=aux_arrays, grad_req=grad_req, shared_exec=base_exec)
    return executor