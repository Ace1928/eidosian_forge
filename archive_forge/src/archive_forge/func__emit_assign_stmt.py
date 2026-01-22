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
def _emit_assign_stmt(lvalue: Union[Constant, Data], rvalue: Data, env: Environment) -> _CodeType:
    if isinstance(lvalue, Constant):
        raise TypeError('lvalue of assignment must not be constant value')
    if isinstance(lvalue.ctype, _cuda_types.Scalar) and isinstance(rvalue.ctype, _cuda_types.Scalar):
        rvalue = _astype_scalar(rvalue, lvalue.ctype, 'same_kind', env)
    elif lvalue.ctype != rvalue.ctype:
        raise TypeError(f'Data type mismatch of variable: `{lvalue.code}`: {lvalue.ctype} != {rvalue.ctype}')
    return [lvalue.ctype.assign(lvalue, rvalue) + ';']