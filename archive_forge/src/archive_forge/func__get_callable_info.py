import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
def _get_callable_info(self, state, expr):

    def get_func_type(state, expr):
        func_ty = None
        if expr.op == 'call':
            try:
                func_ty = state.typemap[expr.func.name]
            except KeyError:
                return None
            if not hasattr(func_ty, 'get_call_type'):
                return None
        elif is_operator_or_getitem(expr):
            func_ty = state.typingctx.resolve_value_type(expr.fn)
        else:
            return None
        return func_ty
    if expr.op == 'call':
        to_inline = None
        try:
            to_inline = state.func_ir.get_definition(expr.func)
        except Exception:
            return None
        if getattr(to_inline, 'op', False) == 'make_function':
            return None
    func_ty = get_func_type(state, expr)
    if func_ty is None:
        return None
    sig = state.calltypes[expr]
    if not sig:
        return None
    templates, arg_typs, is_method = (None, None, False)
    if getattr(func_ty, 'template', None) is not None:
        is_method = True
        templates = [func_ty.template]
        arg_typs = (func_ty.template.this,) + sig.args
    else:
        templates = getattr(func_ty, 'templates', None)
        arg_typs = sig.args
    return (templates, sig, arg_typs, is_method)