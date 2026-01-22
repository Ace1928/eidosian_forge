import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_popc(ConcreteTemplate):
    """
    Supported types from `llvm.popc`
    [here](http://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#bit-manipulations-intrinics)
    """
    key = cuda.popc
    cases = [signature(types.int8, types.int8), signature(types.int16, types.int16), signature(types.int32, types.int32), signature(types.int64, types.int64), signature(types.uint8, types.uint8), signature(types.uint16, types.uint16), signature(types.uint32, types.uint32), signature(types.uint64, types.uint64)]