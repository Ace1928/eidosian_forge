from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def test_numba_jit_expanded_function():
    jit_fn = jit(nopython=True, nogil=True)(expand_varargs(2)(function_with_vararg_call_numba))
    assert function_with_vararg_call_numba(1, 2, 3, 4) == jit_fn(1, 2, 3, 4)