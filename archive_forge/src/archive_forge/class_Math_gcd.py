import math
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
@infer_global(math.gcd)
class Math_gcd(ConcreteTemplate):
    cases = [signature(types.int64, types.int64, types.int64), signature(types.int32, types.int32, types.int32), signature(types.int16, types.int16, types.int16), signature(types.int8, types.int8, types.int8), signature(types.uint64, types.uint64, types.uint64), signature(types.uint32, types.uint32, types.uint32), signature(types.uint16, types.uint16, types.uint16), signature(types.uint8, types.uint8, types.uint8)]