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