from typing import Dict, List, Optional, Sequence, Tuple
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.code_template import CodeTemplate
from torchgen.context import with_native_function
from torchgen.model import (
from torchgen.utils import FileManager
from .context import with_native_function_with_differentiability_info
from .gen_trace_type import (
@with_native_function
def unpack_args(f: NativeFunction) -> Tuple[List[str], List[Binding]]:
    body: List[str] = []
    unpacked_bindings: List[Binding] = []
    bindings = [r for a in f.func.schema_order_arguments() for r in cpp.argument(a, method=False, symint=True, cpp_no_default_args=set(), faithful=False, has_tensor_options=False)]
    for i, binding in enumerate(bindings):
        assert not isinstance(binding.argument, SelfArgument)
        if isinstance(binding.argument, TensorOptionsArguments):
            raise RuntimeError("VariableKernel shouldn't take TensorOptions")
        is_nullable = binding.argument.type.is_nullable()
        if not binding.argument.type.is_tensor_like() or is_nullable:
            unpacked_bindings.append(binding)
            continue
        is_tensor_list = is_tensor_list_type(binding.argument.type)
        ref = not is_nullable and (not is_tensor_list)
        suffix = '_opt' if is_nullable and (not is_tensor_list) else ''
        body.append(UNPACK_TENSOR.substitute(arg_name=binding.name, arg_pos=i, suffix=suffix, ref='&' if ref else ''))
        unpacked_bindings.append(Binding(name=unpacked_name(binding.name), nctype=binding.nctype, argument=binding.argument, default=binding.default))
    return (body, unpacked_bindings)