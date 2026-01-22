import math
from numba.core import types
from numba.core.typing.templates import ConcreteTemplate, signature, Registry
@infer_global(math.sin)
@infer_global(math.cos)
@infer_global(math.ceil)
@infer_global(math.floor)
@infer_global(math.sqrt)
@infer_global(math.log)
@infer_global(math.log2)
@infer_global(math.log10)
@infer_global(math.exp)
@infer_global(math.fabs)
@infer_global(math.trunc)
class Math_unary_with_fp16(ConcreteTemplate):
    cases = [signature(types.float64, types.int64), signature(types.float64, types.uint64), signature(types.float32, types.float32), signature(types.float64, types.float64), signature(types.float16, types.float16)]