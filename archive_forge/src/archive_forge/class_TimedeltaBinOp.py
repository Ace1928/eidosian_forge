from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
class TimedeltaBinOp(AbstractTemplate):

    def generic(self, args, kws):
        if len(args) == 1:
            return
        left, right = args
        if not all((isinstance(tp, types.NPTimedelta) for tp in args)):
            return
        if npdatetime_helpers.can_cast_timedelta_units(left.unit, right.unit):
            return signature(right, left, right)
        elif npdatetime_helpers.can_cast_timedelta_units(right.unit, left.unit):
            return signature(left, left, right)