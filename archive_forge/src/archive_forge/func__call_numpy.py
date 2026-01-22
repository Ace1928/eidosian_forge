import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _call_numpy(func, args, kwargs):
    """
    Calls numpy function with *args and **kwargs and
    does necessary data transfers.

    Args:
        func: A numpy function that needs to be called.
        args (tuple): Arguments.
        kwargs (dict): Keyword arguments.

    Returns:
        Result after calling func and performing data transfers.
    """
    _update_numpy_args(args, kwargs)
    numpy_args, numpy_kwargs = _convert_fallback_to_numpy(args, kwargs)
    numpy_res = func(*numpy_args, **numpy_kwargs)
    ext_res = _get_same_reference(numpy_res, numpy_args, numpy_kwargs, args, kwargs)
    if ext_res is not None:
        return ext_res
    if isinstance(numpy_res, np.ndarray):
        if numpy_res.base is None:
            fallback_res = _convert_numpy_to_fallback(numpy_res)
        else:
            base_arg = _get_same_reference(numpy_res.base, numpy_args, numpy_kwargs, args, kwargs)
            fallback_res = _convert_numpy_to_fallback(numpy_res)
            fallback_res.base = base_arg
        return fallback_res
    return numpy_res