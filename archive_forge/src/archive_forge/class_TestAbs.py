import contextlib
import sys
import warnings
import itertools
import operator
import platform
from numpy._utils import _pep440
import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from
from hypothesis.extra import numpy as hynp
import numpy as np
from numpy.testing import (
class TestAbs:

    def _test_abs_func(self, absfunc, test_dtype):
        x = test_dtype(-1.5)
        assert_equal(absfunc(x), 1.5)
        x = test_dtype(0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)
        x = test_dtype(-0.0)
        res = absfunc(x)
        assert_equal(res, 0.0)
        x = test_dtype(np.finfo(test_dtype).max)
        assert_equal(absfunc(x), x.real)
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            x = test_dtype(np.finfo(test_dtype).tiny)
            assert_equal(absfunc(x), x.real)
        x = test_dtype(np.finfo(test_dtype).min)
        assert_equal(absfunc(x), -x.real)

    @pytest.mark.parametrize('dtype', floating_types + complex_floating_types)
    def test_builtin_abs(self, dtype):
        if sys.platform == 'cygwin' and dtype == np.clongdouble and (_pep440.parse(platform.release().split('-')[0]) < _pep440.Version('3.3.0')):
            pytest.xfail(reason='absl is computed in double precision on cygwin < 3.3')
        self._test_abs_func(abs, dtype)

    @pytest.mark.parametrize('dtype', floating_types + complex_floating_types)
    def test_numpy_abs(self, dtype):
        if sys.platform == 'cygwin' and dtype == np.clongdouble and (_pep440.parse(platform.release().split('-')[0]) < _pep440.Version('3.3.0')):
            pytest.xfail(reason='absl is computed in double precision on cygwin < 3.3')
        self._test_abs_func(np.abs, dtype)