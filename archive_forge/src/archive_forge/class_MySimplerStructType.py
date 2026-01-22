import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
@structref.register
class MySimplerStructType(types.StructRef):
    """
    Test associated with this type represent the lowest level uses of structref.
    """
    pass