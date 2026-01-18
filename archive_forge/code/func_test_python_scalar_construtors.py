import operator
from numpy.testing import assert_raises, suppress_warnings
import numpy as np
import pytest
from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
def test_python_scalar_construtors():
    b = asarray(False)
    i = asarray(0)
    f = asarray(0.0)
    c = asarray(0j)
    assert bool(b) == False
    assert int(i) == 0
    assert float(f) == 0.0
    assert operator.index(i) == 0
    assert_raises(TypeError, lambda: bool(asarray([False])))
    assert_raises(TypeError, lambda: int(asarray([0])))
    assert_raises(TypeError, lambda: float(asarray([0.0])))
    assert_raises(TypeError, lambda: complex(asarray([0j])))
    assert_raises(TypeError, lambda: operator.index(asarray([0])))
    assert bool(b) is bool(i) is bool(f) is bool(c) is False
    assert int(b) == int(i) == int(f) == 0
    assert_raises(TypeError, lambda: int(c))
    assert float(b) == float(i) == float(f) == 0.0
    assert_raises(TypeError, lambda: float(c))
    assert complex(b) == complex(i) == complex(f) == complex(c) == 0j
    assert operator.index(i) == 0
    assert_raises(TypeError, lambda: operator.index(b))
    assert_raises(TypeError, lambda: operator.index(f))
    assert_raises(TypeError, lambda: operator.index(c))