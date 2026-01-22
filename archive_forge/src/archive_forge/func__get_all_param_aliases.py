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
@staticmethod
def _get_all_param_aliases() -> Dict[str, List[str]]:
    buffer_len = 1 << 20
    tmp_out_len = ctypes.c_int64(0)
    string_buffer = ctypes.create_string_buffer(buffer_len)
    ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
    _safe_call(_LIB.LGBM_DumpParamAliases(ctypes.c_int64(buffer_len), ctypes.byref(tmp_out_len), ptr_string_buffer))
    actual_len = tmp_out_len.value
    if actual_len > buffer_len:
        string_buffer = ctypes.create_string_buffer(actual_len)
        ptr_string_buffer = ctypes.c_char_p(ctypes.addressof(string_buffer))
        _safe_call(_LIB.LGBM_DumpParamAliases(ctypes.c_int64(actual_len), ctypes.byref(tmp_out_len), ptr_string_buffer))
    return json.loads(string_buffer.value.decode('utf-8'), object_hook=lambda obj: {k: [k] + v for k, v in obj.items()})