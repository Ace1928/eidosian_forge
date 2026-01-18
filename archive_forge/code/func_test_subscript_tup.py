from __future__ import annotations
import sys
import types
from typing import Any
import pytest
import numpy as np
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
@pytest.mark.parametrize('arg_len', range(4))
def test_subscript_tup(self, cls: type[np.ndarray], arg_len: int) -> None:
    arg_tup = (Any,) * arg_len
    if arg_len in (1, 2):
        assert cls[arg_tup]
    else:
        match = f'Too {('few' if arg_len == 0 else 'many')} arguments'
        with pytest.raises(TypeError, match=match):
            cls[arg_tup]