import sys
import warnings
from numpy.testing import assert_, assert_equal, IS_PYPY
import pytest
from pytest import raises as assert_raises
import scipy.special as sc
from scipy.special._ufuncs import _sf_error_test_function
@pytest.mark.skipif(IS_PYPY, reason='Test not meaningful on PyPy')
def test_sf_error_special_refcount():
    refcount_before = sys.getrefcount(sc)
    with sc.errstate(all='raise'):
        with pytest.raises(sc.SpecialFunctionError, match='domain error'):
            sc.ndtri(2.0)
    refcount_after = sys.getrefcount(sc)
    assert refcount_after == refcount_before