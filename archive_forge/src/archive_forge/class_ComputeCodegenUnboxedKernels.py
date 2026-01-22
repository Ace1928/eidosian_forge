import argparse
import os
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple, Union
import yaml
from torchgen import dest
from torchgen.api import cpp as aten_cpp
from torchgen.api.types import CppSignature, CppSignatureGroup, CType, NamedCType
from torchgen.context import (
from torchgen.executorch.api import et_cpp
from torchgen.executorch.api.custom_ops import (
from torchgen.executorch.api.types import contextArg, ExecutorchCppSignature
from torchgen.executorch.api.unboxing import Unboxing
from torchgen.executorch.model import ETKernelIndex, ETKernelKey, ETParsedYaml
from torchgen.executorch.parse import ET_FIELDS, parse_et_yaml, parse_et_yaml_struct
from torchgen.gen import (
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
@dataclass(frozen=True)
class ComputeCodegenUnboxedKernels:
    selector: SelectiveBuilder
    use_aten_lib: bool

    @method_with_nested_native_function
    def __call__(self, unbox_kernel_entry: Tuple[NativeFunction, Tuple[ETKernelKey, BackendMetadata]]) -> str:
        f: NativeFunction = unbox_kernel_entry[0]
        kernel_key: Union[ETKernelKey, List[ETKernelKey]] = unbox_kernel_entry[1][0]
        kernel_meta: BackendMetadata = unbox_kernel_entry[1][1]
        op_name = f'{f.namespace}::{f.func.name}'
        if not self.selector.is_root_operator(op_name):
            return ''
        if not isinstance(kernel_key, list):
            kernel_key = [kernel_key]
        used_kernel_keys = self.selector.et_get_selected_kernels(op_name, [k.to_native_string() for k in kernel_key])
        if not used_kernel_keys:
            return ''
        sig: Union[CppSignature, ExecutorchCppSignature]
        argument_type_gen: Callable[..., NamedCType]
        return_type_gen: Callable[..., CType]
        if self.use_aten_lib:
            sig = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=f.manual_cpp_binding).most_faithful_signature()
            argument_type_gen = aten_cpp.argumenttype_type
            return_type_gen = aten_cpp.returns_type
            arguments = sig.arguments()
            kernel_call = f'torch::executor::{f.namespace}::{sig.name()}'
        else:
            sig = ExecutorchCppSignature.from_native_function(f)
            argument_type_gen = et_cpp.argumenttype_type
            return_type_gen = et_cpp.returns_type
            arguments = sig.arguments(include_context=False)
            kernel_call = f'{kernel_meta.cpp_namespace}::{kernel_meta.kernel}'
        binding_list, code_list = Unboxing(argument_type_gen=argument_type_gen).convert_arguments(arguments)
        code_connector = '\n\t'
        arg_connector = ', '
        args_str = f'{arg_connector.join((e.name for e in binding_list))}'
        event_tracer_output_logging = ''
        output_ids = []
        if len(f.func.returns) == 0:
            if len(f.func.arguments.out) == 0:
                raise Exception(f"Can't handle native function {f.func} with no returns and no out yet.")
            out = f.func.arguments.out[0]
            return_assignment = f'stack[{len(binding_list)}] = &{out.name};'
            ret_prefix = ''
            output_ids = [len(binding_list)]
        elif len(f.func.arguments.out) == 0:
            return_assignment = f'*stack[{len(binding_list)}] = EValue(result_);'
            ret_prefix = return_type_gen(f.func.returns).cpp_type() + ' result_ = '
            output_ids = [len(binding_list)]
        else:
            return_assignment = ''
            ret_prefix = ''
            output_ids = [len(binding_list) - (i + 1) for i in reversed(range(len(f.func.arguments.out)))]
        for output_id in output_ids:
            event_tracer_output_logging += f'internal::event_tracer_log_evalue(context.internal_event_tracer(), *stack[{output_id}]);\n'
        newline = '\n    '
        return '\n'.join([f'\nKernel(\n    "{f.namespace}::{f.func.name}",{(newline + '"' + (k + '",') if k != 'default' else '')}\n    []({contextArg.defn()}, EValue** stack) {{\n        {code_connector.join(code_list)}\n\n        internal::EventTracerProfileScope event_tracer_scope(context.internal_event_tracer(), "native_call_{f.func.name}");\n        EXECUTORCH_SCOPE_PROF("native_call_{f.func.name}");\n        {ret_prefix}{kernel_call}(context, {args_str});\n        {event_tracer_output_logging}\n        {return_assignment}\n    }}\n),\n' for k in used_kernel_keys])