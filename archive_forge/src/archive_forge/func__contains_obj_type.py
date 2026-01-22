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
def _contains_obj_type(*, pat: Any, checker: Any) -> bool:
    """Determine if the object fits some rule or is array of objects that do so."""
    if isinstance(checker, type):
        targtype = checker
        checker = lambda x: isinstance(x, targtype)
    if checker(pat):
        return True
    if getattr(pat, 'dtype', 'no') != np.object_:
        return False
    return _apply_str_ufunc(func=checker, obj=pat).all()