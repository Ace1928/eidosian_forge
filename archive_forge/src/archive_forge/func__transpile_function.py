import ast
import collections
import inspect
import linecache
import numbers
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union
import warnings
import types
import numpy
from cupy_backends.cuda.api import runtime
from cupy._core._codeblock import CodeBlock, _CodeType
from cupy._core import _kernel
from cupy._core._dtype import _raise_if_invalid_cast
from cupyx import jit
from cupyx.jit import _cuda_types
from cupyx.jit import _cuda_typerules
from cupyx.jit import _internal_types
from cupyx.jit._internal_types import Data
from cupyx.jit._internal_types import Constant
from cupyx.jit import _builtin_funcs
from cupyx.jit import _interface
def _transpile_function(func, name, attributes, mode, consts, in_types, ret_type, generated, *, source):
    """Transpile the function
    Args:
        func (ast.FunctionDef): Target function.
        name (str): Function name.
        attributes (str): The attributes of target function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        consts (dict): The dictionary with keys as variable names and
            values as concrete data object.
        in_types (list of _cuda_types.TypeBase): The types of arguments.
        ret_type (_cuda_types.TypeBase): The type of return value.

    Returns:
        code (str): The generated CUDA code.
        env (Environment): More details of analysis result of the function,
            which includes preambles, estimated return type and more.
    """
    try:
        return _transpile_function_internal(func, name, attributes, mode, consts, in_types, ret_type, generated)
    except _JitCompileError as e:
        exc = e
        if _is_debug_mode:
            exc.reraise(source)
    exc.reraise(source)
    assert False