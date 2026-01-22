import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
@register
class Cuda_shfl_sync_intrinsic(ConcreteTemplate):
    key = cuda.shfl_sync_intrinsic
    cases = [signature(types.Tuple((types.i4, types.b1)), types.i4, types.i4, types.i4, types.i4, types.i4), signature(types.Tuple((types.i8, types.b1)), types.i4, types.i4, types.i8, types.i4, types.i4), signature(types.Tuple((types.f4, types.b1)), types.i4, types.i4, types.f4, types.i4, types.i4), signature(types.Tuple((types.f8, types.b1)), types.i4, types.i4, types.f8, types.i4, types.i4)]