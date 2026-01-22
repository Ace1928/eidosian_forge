import types as pytypes  # avoid confusion with numba.types
import sys, math
import os
import textwrap
import copy
import inspect
import linecache
from functools import reduce
from collections import defaultdict, OrderedDict, namedtuple
from contextlib import contextmanager
import operator
from dataclasses import make_dataclass
import warnings
from llvmlite import ir as lir
from numba.core.imputils import impl_ret_untracked
import numba.core.ir
from numba.core import types, typing, utils, errors, ir, analysis, postproc, rewrites, typeinfer, config, ir_utils
from numba import prange, pndindex
from numba.np.npdatetime_helpers import datetime_minimum, datetime_maximum
from numba.np.numpy_support import as_dtype, numpy_version
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.stencils.stencilparfor import StencilPass
from numba.core.extending import register_jitable, lower_builtin
from numba.core.ir_utils import (
from numba.core.analysis import (compute_use_defs, compute_live_map,
from numba.core.controlflow import CFGraph
from numba.core.typing import npydecl, signature
from numba.core.types.functions import Function
from numba.parfors.array_analysis import (random_int_args, random_1arg_size,
from numba.core.extending import overload
import copy
import numpy
import numpy as np
from numba.parfors import array_analysis
import numba.cpython.builtins
from numba.stencils import stencilparfor
def _replace_parallel_functions(self, blocks):
    """
        Replace functions with their parallel implementation in
        replace_functions_map if available.
        The implementation code is inlined to enable more optimization.
        """
    swapped = self.swapped
    from numba.core.inline_closurecall import inline_closure_call
    work_list = list(blocks.items())
    while work_list:
        label, block = work_list.pop()
        for i, instr in enumerate(block.body):
            if isinstance(instr, ir.Assign):
                lhs = instr.target
                lhs_typ = self.typemap[lhs.name]
                expr = instr.value
                if isinstance(expr, ir.Expr) and expr.op == 'call':

                    def replace_func():
                        func_def = get_definition(self.func_ir, expr.func)
                        callname = find_callname(self.func_ir, expr)
                        repl_func = self.replace_functions_map.get(callname, None)
                        if repl_func is None and len(callname) == 2 and isinstance(callname[1], ir.Var) and isinstance(self.typemap[callname[1].name], types.npytypes.Array):
                            repl_func = replace_functions_ndarray.get(callname[0], None)
                            if repl_func is not None:
                                expr.args.insert(0, callname[1])
                        require(repl_func is not None)
                        typs = tuple((self.typemap[x.name] for x in expr.args))
                        kws_typs = {k: self.typemap[x.name] for k, x in expr.kws}
                        try:
                            new_func = repl_func(lhs_typ, *typs, **kws_typs)
                        except:
                            new_func = None
                        require(new_func is not None)
                        typs = utils.pysignature(new_func).bind(*typs, **kws_typs).args
                        g = copy.copy(self.func_ir.func_id.func.__globals__)
                        g['numba'] = numba
                        g['np'] = numpy
                        g['math'] = math
                        check = replace_functions_checkers_map.get(callname, None)
                        if check is not None:
                            g[check.name] = check.func
                        new_blocks, _ = inline_closure_call(self.func_ir, g, block, i, new_func, self.typingctx, self.targetctx, typs, self.typemap, self.calltypes, work_list)
                        call_table = get_call_table(new_blocks, topological_ordering=False)
                        for call in call_table:
                            for k, v in call.items():
                                if v[0] == 'internal_prange':
                                    swapped[k] = [callname, repl_func.__name__, func_def, block.body[i].loc]
                                    break
                        return True
                    if guard(replace_func):
                        self.stats['replaced_func'] += 1
                        break
                elif isinstance(expr, ir.Expr) and expr.op == 'getattr' and (expr.attr == 'dtype'):
                    typ = self.typemap[expr.value.name]
                    if isinstance(typ, types.npytypes.Array):
                        dtype = typ.dtype
                        scope = block.scope
                        loc = instr.loc
                        g_np_var = ir.Var(scope, mk_unique_var('$np_g_var'), loc)
                        self.typemap[g_np_var.name] = types.misc.Module(numpy)
                        g_np = ir.Global('np', numpy, loc)
                        g_np_assign = ir.Assign(g_np, g_np_var, loc)
                        dtype_str = str(dtype)
                        if dtype_str == 'bool':
                            dtype_str = 'bool_'
                        typ_var = ir.Var(scope, mk_unique_var('$np_typ_var'), loc)
                        self.typemap[typ_var.name] = types.StringLiteral(dtype_str)
                        typ_var_assign = ir.Assign(ir.Const(dtype_str, loc), typ_var, loc)
                        dtype_attr_var = ir.Var(scope, mk_unique_var('$dtype_attr_var'), loc)
                        temp = find_template(numpy.dtype)
                        tfunc = numba.core.types.Function(temp)
                        tfunc.get_call_type(self.typingctx, (self.typemap[typ_var.name],), {})
                        self.typemap[dtype_attr_var.name] = types.functions.Function(temp)
                        dtype_attr_getattr = ir.Expr.getattr(g_np_var, 'dtype', loc)
                        dtype_attr_assign = ir.Assign(dtype_attr_getattr, dtype_attr_var, loc)
                        dtype_var = ir.Var(scope, mk_unique_var('$dtype_var'), loc)
                        self.typemap[dtype_var.name] = types.npytypes.DType(dtype)
                        dtype_getattr = ir.Expr.call(dtype_attr_var, [typ_var], (), loc)
                        dtype_assign = ir.Assign(dtype_getattr, dtype_var, loc)
                        self.calltypes[dtype_getattr] = signature(self.typemap[dtype_var.name], self.typemap[typ_var.name])
                        instr.value = dtype_var
                        block.body.insert(0, dtype_assign)
                        block.body.insert(0, dtype_attr_assign)
                        block.body.insert(0, typ_var_assign)
                        block.body.insert(0, g_np_assign)
                        self.stats['replaced_dtype'] += 1
                        break