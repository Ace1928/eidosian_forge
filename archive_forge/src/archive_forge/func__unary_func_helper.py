import numpy as _np
from ...base import numeric_types, integer_types
from ...util import _sanity_check_params, set_module
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...context import current_context
from . import _internal as _npi
from . import _api_internal
from ..ndarray import NDArray
def _unary_func_helper(x, fn_array, fn_scalar, out=None, **kwargs):
    """Helper function for unary operators.

    Parameters
    ----------
    x : ndarray or scalar
        Input of the unary operator.
    fn_array : function
        Function to be called if x is of ``ndarray`` type.
    fn_scalar : function
        Function to be called if x is a Python scalar.
    out : ndarray
        The buffer ndarray for storing the result of the unary function.

    Returns
    -------
    out : mxnet.numpy.ndarray or scalar
        Result array or scalar.
    """
    if isinstance(x, numeric_types):
        return fn_scalar(x, **kwargs)
    elif isinstance(x, NDArray):
        return fn_array(x, out=out, **kwargs)
    else:
        raise TypeError('type {} not supported'.format(str(type(x))))