from __future__ import annotations
import sys
import types
from typing import Any
import pytest
import numpy as np
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl
class TestSimpleStridedCall:
    method = get_castingimpl(type(np.dtype('d')), type(np.dtype('f')))

    @pytest.mark.parametrize(['args', 'error'], [((True,), TypeError), (((None,),), TypeError), ((None, None), TypeError), (((None, None, None),), TypeError), (((np.arange(3), np.arange(3)),), TypeError), (((np.ones(3, dtype='>d'), np.ones(3, dtype='<f')),), TypeError), (((np.ones((2, 2), dtype='d'), np.ones((2, 2), dtype='f')),), ValueError), (((np.ones(3, dtype='d'), np.ones(4, dtype='f')),), ValueError), (((np.frombuffer(b'\x00x00' * 3 * 2, dtype='d'), np.frombuffer(b'\x00x00' * 3, dtype='f')),), ValueError)])
    def test_invalid_arguments(self, args, error):
        with pytest.raises(error):
            self.method._simple_strided_call(*args)