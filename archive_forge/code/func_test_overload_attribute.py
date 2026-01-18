import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def test_overload_attribute(self):

    @njit
    def check():
        obj = PolygonStruct(5, None)
        return obj.prop[0]
    got = check()
    expect = 5
    self.assertPreciseEqual(got, expect)