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
def _call_ufunc(ufunc: _kernel.ufunc, args: Sequence[Union[Constant, Data]], dtype: Optional[numpy.dtype], env: Environment) -> Data:
    if len(args) != ufunc.nin:
        raise ValueError('invalid number of arguments')
    in_types = []
    for x in args:
        if isinstance(x, Constant):
            t = _cuda_typerules.get_ctype_from_scalar(env.mode, x.obj).dtype
        elif isinstance(x.ctype, _cuda_types.Scalar):
            t = x.ctype.dtype
        else:
            raise TypeError(f'cupy.ufunc: {x.ctype} is unsupported')
        in_types.append(t)
    op = _cuda_typerules.guess_routine(ufunc, tuple(in_types), dtype, env.mode)
    if op is None:
        raise TypeError(f'"{ufunc.name}" does not support for the input types: {in_types}')
    if op.error_func is not None:
        op.error_func()
    if ufunc.nout == 1 and op.routine.startswith('out0 = '):
        out_type = _cuda_types.Scalar(op.out_types[0])
        expr = op.routine.replace('out0 = ', '')
        in_params = []
        for x, t in zip(args, op.in_types):
            x = _astype_scalar(x, _cuda_types.Scalar(t), 'same_kind', env)
            x = Data.init(x, env)
            in_params.append(x)
        can_use_inline_expansion = True
        for i in range(ufunc.nin):
            if len(list(re.finditer('in{}'.format(i), op.routine))) > 1:
                can_use_inline_expansion = False
            if f'in{i}_type' in op.routine:
                can_use_inline_expansion = False
        env.generated.add_code(ufunc._preamble)
        if can_use_inline_expansion:
            for i, x in enumerate(in_params):
                expr = expr.replace(f'in{i}', x.code)
            expr = '(' + expr.replace('out0_type', str(out_type)) + ')'
        else:
            template_typenames = ', '.join([f'typename in{i}_type' for i in range(ufunc.nin)])
            ufunc_name = f'{ufunc.name}_{str(numpy.dtype(op.out_types[0]))}'
            params = ', '.join([f'in{i}_type in{i}' for i in range(ufunc.nin)])
            ufunc_code = f'template <{template_typenames}>\n__device__ {out_type} {ufunc_name}({params}) {{\n    typedef {out_type} out0_type;\n    return {expr};\n}}\n'
            env.generated.add_code(ufunc_code)
            in_params_code = ', '.join([a.code for a in in_params])
            expr = f'{ufunc_name}({in_params_code})'
        return Data(expr, out_type)
    raise NotImplementedError(f'ufunc `{ufunc.name}` is not supported.')