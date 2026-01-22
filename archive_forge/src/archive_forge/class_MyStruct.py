import unittest
import numpy as np
from numba import njit
from numba.core import types
from numba.experimental import structref
from numba.tests.support import skip_unless_scipy
class MyStruct(structref.StructRefProxy):

    def __new__(cls, name, vector):
        return structref.StructRefProxy.__new__(cls, name, vector)

    @property
    def name(self):
        return MyStruct_get_name(self)

    @property
    def vector(self):
        return MyStruct_get_vector(self)