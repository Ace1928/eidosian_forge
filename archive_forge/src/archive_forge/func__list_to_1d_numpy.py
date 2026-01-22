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
def _list_to_1d_numpy(data: Any, dtype: 'np.typing.DTypeLike', name: str) -> np.ndarray:
    """Convert data to numpy 1-D array."""
    if _is_numpy_1d_array(data):
        return _cast_numpy_array_to_dtype(data, dtype)
    elif _is_numpy_column_array(data):
        _log_warning('Converting column-vector to 1d array')
        array = data.ravel()
        return _cast_numpy_array_to_dtype(array, dtype)
    elif _is_1d_list(data):
        return np.array(data, dtype=dtype, copy=False)
    elif isinstance(data, pd_Series):
        _check_for_bad_pandas_dtypes(data.to_frame().dtypes)
        return np.array(data, dtype=dtype, copy=False)
    else:
        raise TypeError(f'Wrong type({type(data).__name__}) for {name}.\nIt should be list, numpy 1-D array or pandas Series')