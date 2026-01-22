import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def _get_cusparselt_torch_version() -> Tuple[int, int, int]:
    """
    Returns the version of the cusparselt.so library that ships with pytorch 2.2+
    """
    lib_path = _get_cusparselt_lib()
    if lib_path is None:
        return (0, 0, 0)
    lib = ctypes.CDLL(lib_path)

    def get_version_part(version_part: int) -> int:
        value = ctypes.c_int()
        ret = lib.cusparseLtGetProperty(version_part, ctypes.byref(value))
        if ret != 0:
            return -1
        return value.value
    return (get_version_part(0), get_version_part(1), get_version_part(2))