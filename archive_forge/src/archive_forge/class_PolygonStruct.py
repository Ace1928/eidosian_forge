import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
class PolygonStruct(structref.StructRefProxy):

    def __new__(cls, value, parent):
        return structref.StructRefProxy.__new__(cls, value, parent)

    @property
    def value(self):
        return PolygonStruct_get_value(self)

    @property
    def parent(self):
        return PolygonStruct_get_parent(self)