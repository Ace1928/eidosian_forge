from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version

        (timedelta64, {int, float}) -> timedelta64
        (timedelta64, timedelta64) -> float
        