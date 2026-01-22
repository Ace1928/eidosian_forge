import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def _arg_func_to_proto(value: ARG_LIKE, arg_function_language: Optional[str], msg: Union[v2.program_pb2.Arg, v2.program_pb2.FloatArg]) -> None:

    def check_support(func_type: str) -> str:
        if func_type not in supported:
            lang = repr(arg_function_language) if arg_function_language is not None else '[any]'
            raise ValueError(f'Function type {func_type!r} not supported by arg_function_language {lang}')
        return func_type
    if arg_function_language not in SUPPORTED_FUNCTIONS_FOR_LANGUAGE:
        raise ValueError(f'Unrecognized arg_function_language: {arg_function_language!r}')
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE[arg_function_language]
    if isinstance(value, sympy.Symbol):
        msg.symbol = str(value.free_symbols.pop())
    elif isinstance(value, sympy.Add):
        msg.func.type = check_support('add')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    elif isinstance(value, sympy.Mul):
        msg.func.type = check_support('mul')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    elif isinstance(value, sympy.Pow):
        msg.func.type = check_support('pow')
        for arg in value.args:
            arg_to_proto(arg, arg_function_language=arg_function_language, out=msg.func.args.add())
    else:
        raise ValueError(f"Unrecognized Sympy expression type: {type(value)}. Only the following types are recognized: 'sympy.Symbol', 'sympy.Add', 'sympy.Mul', 'sympy.Pow'.")