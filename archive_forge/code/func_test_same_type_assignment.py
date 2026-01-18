import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_same_type_assignment(self):

    @njit
    def check(x):
        poly = PolygonStruct(None, None)
        p_poly = PolygonStruct(None, None)
        poly.value = x
        poly.parent = p_poly
        p_poly.value = x
        return poly.parent.value
    x = 11
    got = check(x)
    expect = x
    self.assertPreciseEqual(got, expect)