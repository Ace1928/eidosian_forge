from __future__ import annotations
import functools
import math
from typing import Sequence
import torch
from . import _dtypes_impl, _util
from ._normalizations import ArrayLike, KeepDims, normalizer
def _atleast_float_2(a, b):
    dtyp = _dtypes_impl.result_type_impl(a, b)
    if not (dtyp.is_floating_point or dtyp.is_complex):
        dtyp = _dtypes_impl.default_dtypes().float_dtype
    a = _util.cast_if_needed(a, dtyp)
    b = _util.cast_if_needed(b, dtyp)
    return (a, b)