import types
import numpy as np
import cupy as cp
from cupyx.fallback_mode import notification
def _get_xp_args(ndarray_instance, to_xp, arg):
    """
    Converts ndarray_instance type object to target object using to_xp.

    Args:
        ndarray_instance (numpy.ndarray, cupy.ndarray or fallback.ndarray):
            Objects of type `ndarray_instance` will be converted using `to_xp`.
        to_xp (FunctionType): Method to convert ndarray_instance type objects.
        arg (object): `ndarray_instance`, `tuple`, `list` and `dict` type
            objects will be returned by either converting the object or it's
            elements, if object is iterable. Objects of other types is
            returned as it is.

    Returns:
        Return data structure will be same as before after converting ndarrays.
    """
    if isinstance(arg, ndarray_instance):
        return to_xp(arg)
    if isinstance(arg, tuple):
        return tuple([_get_xp_args(ndarray_instance, to_xp, x) for x in arg])
    if isinstance(arg, dict):
        return {x_name: _get_xp_args(ndarray_instance, to_xp, x) for x_name, x in arg.items()}
    if isinstance(arg, list):
        return [_get_xp_args(ndarray_instance, to_xp, x) for x in arg]
    return arg