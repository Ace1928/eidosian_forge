import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.frexp)
class Math_frexp(ConcreteTemplate):
    cases = [signature(types.Tuple((types.float64, types.intc)), types.float64), signature(types.Tuple((types.float32, types.intc)), types.float32)]