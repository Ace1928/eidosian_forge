import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@structref.register
class PolygonStructType(types.StructRef):

    def preprocess_fields(self, fields):
        self.name = f'numba.PolygonStructType#{id(self)}'
        fields = tuple([('value', types.Optional(types.int64)), ('parent', types.Optional(self))])
        return fields