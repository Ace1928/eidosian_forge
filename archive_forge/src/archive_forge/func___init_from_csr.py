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
def __init_from_csr(self, csr: scipy.sparse.csr_matrix, params_str: str, ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """Initialize data from a CSR matrix."""
    if len(csr.indices) != len(csr.data):
        raise ValueError(f'Length mismatch: {len(csr.indices)} vs {len(csr.data)}')
    self._handle = ctypes.c_void_p()
    ptr_indptr, type_ptr_indptr, __ = _c_int_array(csr.indptr)
    ptr_data, type_ptr_data, _ = _c_float_array(csr.data)
    assert csr.shape[1] <= _MAX_INT32
    csr_indices = csr.indices.astype(np.int32, copy=False)
    _safe_call(_LIB.LGBM_DatasetCreateFromCSR(ptr_indptr, ctypes.c_int(type_ptr_indptr), csr_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), ptr_data, ctypes.c_int(type_ptr_data), ctypes.c_int64(len(csr.indptr)), ctypes.c_int64(len(csr.data)), ctypes.c_int64(csr.shape[1]), _c_str(params_str), ref_dataset, ctypes.byref(self._handle)))
    return self