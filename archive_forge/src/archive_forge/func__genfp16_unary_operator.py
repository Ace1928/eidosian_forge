import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
def _genfp16_unary_operator(l_key):

    @register_global(l_key)
    class Cuda_fp16_unary(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) == 1 and args[0] == types.float16:
                return signature(types.float16, types.float16)
    return Cuda_fp16_unary