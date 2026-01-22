import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def _arg_func_from_proto(func: v2.program_pb2.ArgFunction, *, arg_function_language: str, required_arg_name: Optional[str]=None) -> Optional[ARG_RETURN_LIKE]:
    supported = SUPPORTED_FUNCTIONS_FOR_LANGUAGE.get(arg_function_language)
    if supported is None:
        raise ValueError(f'Unrecognized arg_function_language: {arg_function_language!r}')
    if func.type not in supported:
        raise ValueError(f'Unrecognized function type {func.type!r} for arg_function_language={arg_function_language!r}')
    if func.type == 'add':
        return sympy.Add(*[arg_from_proto(a, arg_function_language=arg_function_language, required_arg_name='An addition argument') for a in func.args])
    if func.type == 'mul':
        return sympy.Mul(*[arg_from_proto(a, arg_function_language=arg_function_language, required_arg_name='A multiplication argument') for a in func.args])
    if func.type == 'pow':
        return sympy.Pow(*[arg_from_proto(a, arg_function_language=arg_function_language, required_arg_name='A power argument') for a in func.args])
    return None