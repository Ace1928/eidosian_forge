import abc
import ctypes
import inspect
import json
import warnings
from collections import OrderedDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from os import SEEK_END, environ
from os.path import getsize
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Union
import numpy as np
import scipy.sparse
from .compat import (PANDAS_INSTALLED, PYARROW_INSTALLED, arrow_cffi, arrow_is_floating, arrow_is_integer, concat,
from .libpath import find_lib_path
def __init_from_list_np2d(self, mats: List[np.ndarray], params_str: str, ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """Initialize data from a list of 2-D numpy matrices."""
    ncol = mats[0].shape[1]
    nrow = np.empty((len(mats),), np.int32)
    ptr_data: _ctypes_float_array
    if mats[0].dtype == np.float64:
        ptr_data = (ctypes.POINTER(ctypes.c_double) * len(mats))()
    else:
        ptr_data = (ctypes.POINTER(ctypes.c_float) * len(mats))()
    holders = []
    type_ptr_data = -1
    for i, mat in enumerate(mats):
        if len(mat.shape) != 2:
            raise ValueError('Input numpy.ndarray must be 2 dimensional')
        if mat.shape[1] != ncol:
            raise ValueError('Input arrays must have same number of columns')
        nrow[i] = mat.shape[0]
        if mat.dtype == np.float32 or mat.dtype == np.float64:
            mats[i] = np.array(mat.reshape(mat.size), dtype=mat.dtype, copy=False)
        else:
            mats[i] = np.array(mat.reshape(mat.size), dtype=np.float32)
        chunk_ptr_data, chunk_type_ptr_data, holder = _c_float_array(mats[i])
        if type_ptr_data != -1 and chunk_type_ptr_data != type_ptr_data:
            raise ValueError('Input chunks must have same type')
        ptr_data[i] = chunk_ptr_data
        type_ptr_data = chunk_type_ptr_data
        holders.append(holder)
    self._handle = ctypes.c_void_p()
    _safe_call(_LIB.LGBM_DatasetCreateFromMats(ctypes.c_int32(len(mats)), ctypes.cast(ptr_data, ctypes.POINTER(ctypes.POINTER(ctypes.c_double))), ctypes.c_int(type_ptr_data), nrow.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ctypes.c_int32(ncol), ctypes.c_int(_C_API_IS_ROW_MAJOR), _c_str(params_str), ref_dataset, ctypes.byref(self._handle)))
    return self