from __future__ import annotations
import abc
from collections import defaultdict
import functools
from functools import partial
import inspect
from typing import (
import warnings
import numpy as np
from pandas._config import option_context
from pandas._libs import lib
from pandas._libs.internals import BlockValuesRefs
from pandas._typing import (
from pandas.compat._optional import import_optional_dependency
from pandas.errors import SpecificationError
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.cast import is_nested_object
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.generic import (
from pandas.core._numba.executor import generate_apply_looper
import pandas.core.common as com
from pandas.core.construction import ensure_wrapped_if_datetimelike
def _apply_str(self, obj, func: str, *args, **kwargs):
    """
        if arg is a string, then try to operate on it:
        - try to find a function (or attribute) on obj
        - try to find a numpy function
        - raise
        """
    assert isinstance(func, str)
    if hasattr(obj, func):
        f = getattr(obj, func)
        if callable(f):
            return f(*args, **kwargs)
        assert len(args) == 0
        assert len([kwarg for kwarg in kwargs if kwarg not in ['axis']]) == 0
        return f
    elif hasattr(np, func) and hasattr(obj, '__array__'):
        f = getattr(np, func)
        return f(obj, *args, **kwargs)
    else:
        msg = f"'{func}' is not a valid function for '{type(obj).__name__}' object"
        raise AttributeError(msg)