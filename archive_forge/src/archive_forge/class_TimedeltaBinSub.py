from itertools import product
import operator
from numba.core import types
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.np import npdatetime_helpers
from numba.np.numpy_support import numpy_version
@infer_global(operator.sub)
@infer_global(operator.isub)
class TimedeltaBinSub(TimedeltaBinOp):
    key = operator.sub