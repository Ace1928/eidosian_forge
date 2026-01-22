import copy
import ctypes
import importlib.util
import json
import os
import re
import sys
import warnings
import weakref
from abc import ABC, abstractmethod
from collections.abc import Mapping
from enum import IntEnum, unique
from functools import wraps
from inspect import Parameter, signature
from typing import (
import numpy as np
import scipy.sparse
from ._typing import (
from .compat import PANDAS_INSTALLED, DataFrame, py_str
from .libpath import find_lib_path
def _init_from_iter(self, iterator: DataIter, enable_categorical: bool) -> None:
    it = iterator
    args = {'missing': self.missing, 'nthread': self.nthread, 'cache_prefix': it.cache_prefix if it.cache_prefix else ''}
    args_cstr = from_pystr_to_cstr(json.dumps(args))
    handle = ctypes.c_void_p()
    reset_callback, next_callback = it.get_callbacks(True, enable_categorical)
    ret = _LIB.XGDMatrixCreateFromCallback(None, it.proxy.handle, reset_callback, next_callback, args_cstr, ctypes.byref(handle))
    it.reraise()
    _check_call(ret)
    self.handle = handle