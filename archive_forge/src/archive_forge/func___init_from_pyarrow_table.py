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
def __init_from_pyarrow_table(self, table: pa_Table, params_str: str, ref_dataset: Optional[_DatasetHandle]) -> 'Dataset':
    """Initialize data from a PyArrow table."""
    if not PYARROW_INSTALLED:
        raise LightGBMError('Cannot init dataframe from Arrow without `pyarrow` installed.')
    if not all((arrow_is_integer(t) or arrow_is_floating(t) for t in table.schema.types)):
        raise ValueError('Arrow table may only have integer or floating point datatypes')
    c_array = _export_arrow_to_c(table)
    self._handle = ctypes.c_void_p()
    _safe_call(_LIB.LGBM_DatasetCreateFromArrow(ctypes.c_int64(c_array.n_chunks), ctypes.c_void_p(c_array.chunks_ptr), ctypes.c_void_p(c_array.schema_ptr), _c_str(params_str), ref_dataset, ctypes.byref(self._handle)))
    return self