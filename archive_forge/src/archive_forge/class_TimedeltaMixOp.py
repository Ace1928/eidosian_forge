from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
class TimedeltaMixOp(AbstractTemplate):

    def generic(self, args, kws):
        """
        (timedelta64, {int, float}) -> timedelta64
        ({int, float}, timedelta64) -> timedelta64
        """
        left, right = args
        if isinstance(right, types.NPTimedelta):
            td, other = (right, left)
            sig_factory = lambda other: signature(td, other, td)
        elif isinstance(left, types.NPTimedelta):
            td, other = (left, right)
            sig_factory = lambda other: signature(td, td, other)
        else:
            return
        if not isinstance(other, (types.Float, types.Integer)):
            return
        if isinstance(other, types.Integer):
            other = types.int64
        return sig_factory(other)