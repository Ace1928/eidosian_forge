from __future__ import annotations
import warnings
import pytest
from datashader.macros import expand_varargs
import inspect
from numba import jit
def test_unsupported_vararg_use():
    with pytest.raises(ValueError) as e:
        expand_varargs(2)(function_with_unsupported_vararg_use)
    assert e.match('unsupported context')