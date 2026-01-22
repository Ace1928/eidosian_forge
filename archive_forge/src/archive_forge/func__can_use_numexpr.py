from __future__ import annotations
import operator
from typing import TYPE_CHECKING
import warnings
import numpy as np
from pandas._config import get_option
from pandas.util._exceptions import find_stack_level
from pandas.core import roperator
from pandas.core.computation.check import NUMEXPR_INSTALLED
def _can_use_numexpr(op, op_str, a, b, dtype_check) -> bool:
    """return a boolean if we WILL be using numexpr"""
    if op_str is not None:
        if a.size > _MIN_ELEMENTS:
            dtypes: set[str] = set()
            for o in [a, b]:
                if hasattr(o, 'dtype'):
                    dtypes |= {o.dtype.name}
            if not len(dtypes) or _ALLOWED_DTYPES[dtype_check] >= dtypes:
                return True
    return False