import ctypes
import functools
import inspect
import threading
from .base import _LIB, check_call, c_str, py_str
def _sanity_check_params(func_name, unsupported_params, param_dict):
    for param_name in unsupported_params:
        if param_name in param_dict:
            raise NotImplementedError('function {} does not support parameter {}'.format(func_name, param_name))