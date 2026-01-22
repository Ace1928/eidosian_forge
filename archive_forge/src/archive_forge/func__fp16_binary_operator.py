import operator
from numba.core import types
from numba.core.typing.npydecl import (parse_dtype, parse_shape,
from numba.core.typing.templates import (AttributeTemplate, ConcreteTemplate,
from numba.cuda.types import dim3
from numba.core.typeconv import Conversion
from numba import cuda
from numba.cuda.compiler import declare_device_function_template
def _fp16_binary_operator(l_key, retty):

    @register_global(l_key)
    class Cuda_fp16_operator(AbstractTemplate):
        key = l_key

        def generic(self, args, kws):
            assert not kws
            if len(args) == 2 and (args[0] == types.float16 or args[1] == types.float16):
                if args[0] == types.float16:
                    convertible = self.context.can_convert(args[1], args[0])
                else:
                    convertible = self.context.can_convert(args[0], args[1])
                if convertible == Conversion.exact or convertible == Conversion.promote or convertible == Conversion.safe:
                    return signature(retty, types.float16, types.float16)
    return Cuda_fp16_operator