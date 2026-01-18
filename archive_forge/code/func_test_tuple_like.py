import pytest
import pickle
from numpy.testing import assert_equal
from scipy._lib._bunch import _make_tuple_bunch
def test_tuple_like(self):
    Tup = _make_tuple_bunch('Tup', ['a', 'b'])
    tu = Tup(a=1, b=2)
    assert isinstance(tu, tuple)
    assert isinstance(tu + (1,), tuple)