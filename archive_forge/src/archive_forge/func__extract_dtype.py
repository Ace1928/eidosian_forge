from __future__ import annotations
import builtins
import math
import operator
from typing import Sequence
import torch
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
def _extract_dtype(entry):
    try:
        dty = _dtypes.dtype(entry)
    except Exception:
        dty = asarray(entry).dtype
    return dty