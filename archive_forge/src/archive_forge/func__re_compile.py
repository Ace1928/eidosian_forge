from __future__ import annotations
import codecs
import re
import textwrap
from collections.abc import Hashable, Mapping
from functools import reduce
from operator import or_ as set_union
from re import Pattern
from typing import TYPE_CHECKING, Any, Callable, Generic
from unicodedata import normalize
import numpy as np
from xarray.core import duck_array_ops
from xarray.core.computation import apply_ufunc
from xarray.core.types import T_DataArray
def _re_compile(self, *, pat: str | bytes | Pattern | Any, flags: int=0, case: bool | None=None) -> Pattern | Any:
    is_compiled_re = isinstance(pat, re.Pattern)
    if is_compiled_re and flags != 0:
        raise ValueError('Flags cannot be set when pat is a compiled regex.')
    if is_compiled_re and case is not None:
        raise ValueError('Case cannot be set when pat is a compiled regex.')
    if is_compiled_re:
        return re.compile(pat)
    if case is None:
        case = True
    if not case:
        flags |= re.IGNORECASE
    if getattr(pat, 'dtype', None) != np.object_:
        pat = self._stringify(pat)

    def func(x):
        return re.compile(x, flags=flags)
    if isinstance(pat, np.ndarray):
        func_ = np.vectorize(func)
        return func_(pat)
    else:
        return _apply_str_ufunc(func=func, obj=pat, dtype=np.object_)