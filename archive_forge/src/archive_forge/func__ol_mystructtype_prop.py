import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@overload_attribute(MyStructType, 'prop')
def _ol_mystructtype_prop(self):

    def get(self):
        return (self.values, self.counter)
    return get