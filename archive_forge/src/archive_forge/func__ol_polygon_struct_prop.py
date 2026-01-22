import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@overload_attribute(PolygonStructType, 'prop')
def _ol_polygon_struct_prop(self):

    def get(self):
        return (self.value, self.parent)
    return get