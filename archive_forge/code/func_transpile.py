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
def transpile(func, attributes, mode, in_types, ret_type):
    """Transpiles the target function.

    Args:
        func (function): Target function.
        attributes (list of str): Attributes of the generated CUDA function.
        mode ('numpy' or 'cuda'): The rule for typecast.
        in_types (list of _cuda_types.TypeBase): Types of the arguments.
        ret_type (_cuda_types.TypeBase or None): Type of the return value.
    """
    generated = Generated()
    in_types = tuple(in_types)
    name, return_type = _transpile_func_obj(func, attributes, mode, in_types, ret_type, generated)
    func_name, _ = generated.device_function[func, in_types]
    code = '\n'.join(generated.codes)
    backend = generated.backend
    enable_cg = generated.enable_cg
    if _is_debug_mode:
        print(code)
    return Result(func_name=func_name, code=code, return_type=return_type, enable_cooperative_groups=enable_cg, backend=backend)