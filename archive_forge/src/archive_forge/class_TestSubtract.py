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
class TestSubtract:

    def test_exceptions(self):
        a = np.ones((), dtype=np.bool_)[()]
        assert_raises(TypeError, operator.sub, a, a)

    def test_result(self):
        types = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning)
            for dt in types:
                a = np.ones((), dtype=dt)[()]
                assert_equal(operator.sub(a, a), 0)