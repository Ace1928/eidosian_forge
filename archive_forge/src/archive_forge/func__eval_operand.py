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
def _eval_operand(op: ast.AST, args: Sequence[Union[Constant, Data]], env: Environment) -> Union[Constant, Data]:
    if is_constants(*args):
        pyfunc = _cuda_typerules.get_pyfunc(type(op))
        return Constant(pyfunc(*[x.obj for x in args]))
    if isinstance(op, ast.Add):
        x, y = args
        x = Data.init(x, env)
        y = Data.init(y, env)
        if hasattr(x.ctype, '_add'):
            out = x.ctype._add(env, x, y)
            if out is not NotImplemented:
                return out
        if hasattr(y.ctype, '_radd'):
            out = y.ctype._radd(env, x, y)
            if out is not NotImplemented:
                return out
    if isinstance(op, ast.Sub):
        x, y = args
        x = Data.init(x, env)
        y = Data.init(y, env)
        if hasattr(x.ctype, '_sub'):
            out = x.ctype._sub(env, x, y)
            if out is not NotImplemented:
                return out
        if hasattr(y.ctype, '_rsub'):
            out = y.ctype._rsub(env, x, y)
            if out is not NotImplemented:
                return out
    ufunc = _cuda_typerules.get_ufunc(env.mode, type(op))
    return _call_ufunc(ufunc, args, None, env)