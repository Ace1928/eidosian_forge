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
@transpile_function_wrapper
def _transpile_expr(expr: ast.expr, env: Environment) -> _internal_types.Expr:
    """Transpile the statement.

    Returns (Data): The CUDA code and its type of the expression.
    """
    res = _transpile_expr_internal(expr, env)
    if isinstance(res, Constant) and isinstance(res.obj, _internal_types.Expr):
        return res.obj
    else:
        return res