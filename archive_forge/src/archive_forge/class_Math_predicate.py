import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.isinf)
@infer_global(math.isnan)
class Math_predicate(ConcreteTemplate):
    cases = [signature(types.boolean, types.int64), signature(types.boolean, types.uint64), signature(types.boolean, types.float32), signature(types.boolean, types.float64)]