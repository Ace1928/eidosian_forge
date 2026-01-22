from typing import Any, TypeVar, Optional, Tuple, List, NamedTuple, Union, Sequence, Dict, Callable
import textwrap
import torch
from torch._C import TupleType, ListType
from torch.jit._recursive import wrap_cpp_module
def _get_bundled_inputs_attributes_and_methods(script_module: torch.jit.ScriptModule) -> Tuple[List[str], List[str]]:
    methods: List[str] = []
    attributes: List[str] = []
    if hasattr(script_module, 'get_all_bundled_inputs'):
        methods.append('get_all_bundled_inputs')
        methods.append('get_num_bundled_inputs')
        methods.append('run_on_bundled_input')
    if hasattr(script_module, 'get_bundled_inputs_functions_and_info'):
        methods.append('get_bundled_inputs_functions_and_info')
        all_info = script_module.get_bundled_inputs_functions_and_info()
        for function_name in all_info:
            methods.append('get_all_bundled_inputs_for_' + function_name)
            methods.append('_generate_bundled_inputs_for_' + function_name)
            attributes.append('_bundled_inputs_deflated_' + function_name)
            bundled_inputs_fn = getattr(script_module, f'get_all_bundled_inputs_for_{function_name}')
            num_bundled_inputs: int = len(bundled_inputs_fn())
            func = getattr(script_module, function_name)
            for arg_idx in range(len(func.schema.arguments) - 1):
                for input_idx in range(num_bundled_inputs):
                    helper_fn_name = _get_inflate_helper_fn_name(arg_idx=arg_idx, input_idx=input_idx, function_name=function_name)
                    if hasattr(script_module, helper_fn_name):
                        methods.append(helper_fn_name)
    return (methods, attributes)