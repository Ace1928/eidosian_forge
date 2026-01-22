import re
import sys
import pytest
import numpy as np
import numpy.core._multiarray_tests as mt
from numpy.testing import assert_warns, IS_PYPY
class TestIntpConverter:
    """ Tests of PyArray_IntpConverter """
    conv = mt.run_intp_converter

    def test_basic(self):
        assert self.conv(1) == (1,)
        assert self.conv((1, 2)) == (1, 2)
        assert self.conv([1, 2]) == (1, 2)
        assert self.conv(()) == ()

    def test_none(self):
        with pytest.warns(DeprecationWarning):
            assert self.conv(None) == ()

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8), reason='PyPy bug in error formatting')
    def test_float(self):
        with pytest.raises(TypeError):
            self.conv(1.0)
        with pytest.raises(TypeError):
            self.conv([1, 1.0])

    def test_too_large(self):
        with pytest.raises(ValueError):
            self.conv(2 ** 64)

    def test_too_many_dims(self):
        assert self.conv([1] * 32) == (1,) * 32
        with pytest.raises(ValueError):
            self.conv([1] * 33)