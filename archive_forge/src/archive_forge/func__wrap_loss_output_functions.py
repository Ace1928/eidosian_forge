from array import array
import ctypes
import logging
import contextlib
import numpy as np
from ... import symbol
from ...context import gpu
from ...symbol import Symbol
from ...module import BucketingModule
from ...symbol import contrib as symbol_contrib
from ... import ndarray
from ...ndarray import NDArray, _DTYPE_NP_TO_MX, _DTYPE_MX_TO_NP
from . import lists
from ...gluon import trainer
from ... import base
from ...base import c_str_array, SymbolHandle, check_call, _LIB, mx_uint, c_array_buf
from ... import optimizer as opt
from .loss_scaler import LossScaler
def _wrap_loss_output_functions(module, ls, target_dtype):
    if module == ndarray:

        def _wrapper(f):

            def _scaling_wrapper(*args, **kwargs):
                if 'grad_scale' in kwargs:
                    kwargs['grad_scale'] = kwargs['grad_scale'] * ls.loss_scale
                else:
                    kwargs['grad_scale'] = ls.loss_scale
                return f(*args, **kwargs)
            _scaling_wrapper.__name__ = f.__name__
            _scaling_wrapper.__module__ = f.__module__
            _scaling_wrapper.__doc__ = f.__doc__
            return _scaling_wrapper
    else:

        def _wrapper(f):

            def _warning_wrapper(*args, **kwargs):
                logging.warning('%s does not support dynamic loss scaling in symbolic and hybridized execution.', f.__name__)
                return f(*args, **kwargs)
            _warning_wrapper.__name__ = f.__name__
            _warning_wrapper.__module__ = f.__module__
            _warning_wrapper.__doc__ = f.__doc__
            return _warning_wrapper
    for fun_name in list_loss_output_functions(target_dtype):
        try:
            f_to_wrap = getattr(module, fun_name)
            setattr(module, fun_name, _wrapper(f_to_wrap))
        except AttributeError:
            pass