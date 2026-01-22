import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.ldexp)
class Math_ldexp(ConcreteTemplate):
    cases = [signature(types.float64, types.float64, types.intc), signature(types.float32, types.float32, types.intc)]