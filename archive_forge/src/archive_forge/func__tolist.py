from __future__ import annotations
import builtins
import math
import operator
from typing import Sequence
import torch
from . import _dtypes, _dtypes_impl, _funcs, _ufuncs, _util
from ._normalizations import (
def _tolist(obj):
    """Recursively convert tensors into lists."""
    a1 = []
    for elem in obj:
        if isinstance(elem, (list, tuple)):
            elem = _tolist(elem)
        if isinstance(elem, ndarray):
            a1.append(elem.tensor.tolist())
        else:
            a1.append(elem)
    return a1