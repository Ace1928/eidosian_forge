import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
class ParametrizedType(types.Type):
    """this is essentially UniTuple(unicode_type, n)
    BUT type name is the same for all n"""

    def __init__(self, value):
        super(ParametrizedType, self).__init__('ParametrizedType')
        self.dtype = types.unicode_type
        self.n = len(value)

    @property
    def key(self):
        return self.n

    def __len__(self):
        return self.n