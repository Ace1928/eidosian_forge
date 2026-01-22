from array import array
import ctypes
import warnings
from numbers import Number
import numpy as _numpy  # pylint: disable=relative-import
from ..attribute import AttrScope
from ..base import _LIB, numeric_types, c_array, c_array_buf, c_str, c_str_array, c_handle_array
from ..base import mx_uint, py_str, string_types, integer_types, mx_int, mx_int64
from ..base import NDArrayHandle, ExecutorHandle, SymbolHandle
from ..base import check_call, MXNetError, NotImplementedForSymbol
from ..context import Context, current_context
from ..ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP, _GRAD_REQ_MAP
from ..ndarray.ndarray import _STORAGE_TYPE_STR_TO_ID, _int64_enabled, _SIGNED_INT32_UPPER_LIMIT
from ..ndarray import _ndarray_cls
from ..executor import Executor
from . import _internal
from . import op
from ._internal import SymbolBase, _set_symbol_class
from ..util import is_np_shape
@staticmethod
def _get_ndarray_inputs(arg_key, args, arg_names, allow_missing):
    """Helper function to get NDArray lists handles from various inputs.

        Parameters
        ----------
        arg_key : str
            The name of argument, used for error message.

        args : list of NDArray or dict of str to NDArray
            Input arguments to the symbols.
            If type is list of NDArray, the position is in the same order of arg_names.
            If type is dict of str to NDArray, then it maps the name of arguments
            to the corresponding NDArray,

        args_names : list of string
            List of argument names.

        allow_missing : boolean
            Whether missing argument is allowed.
            When allowed, the missing handle will be set to None(null)

        Returns
        -------
        handles : list of NDArrayHandle
            The positional list of NDArrayHandles generated from input.
        """
    arg_handles = []
    arg_arrays = []
    if isinstance(args, list):
        if len(args) != len(arg_names):
            raise ValueError('Length of %s does not match the number of arguments' % arg_key)
        for narr in args:
            if narr is None and allow_missing:
                arg_handles.append(None)
            elif not isinstance(narr, NDArray):
                raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
            else:
                arg_handles.append(narr.handle)
        arg_arrays = args
    elif isinstance(args, dict):
        for name in arg_names:
            if name in args:
                narr = args[name]
                if not isinstance(narr, NDArray):
                    raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
                arg_handles.append(narr.handle)
                arg_arrays.append(narr)
            elif allow_missing:
                arg_handles.append(None)
                arg_arrays.append(None)
            else:
                raise ValueError('key `%s` is missing in `%s`' % (name, arg_key))
    else:
        raise TypeError('Only accept list of NDArrays or dict of str to NDArray')
    return (c_array(NDArrayHandle, arg_handles), arg_arrays)