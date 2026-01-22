import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
def _has_cusparseLt() -> bool:
    available = _cusplt_version >= (0, 4, 0)
    if available and _cusplt_version < (0, 5, 0):
        warnings.warn(f'You have cusparseLt version {_cusplt_version_str} but you get better performance with v0.5.0+ if you replace the .so file ({_get_cusparselt_lib()})')
    return available