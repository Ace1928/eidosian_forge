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
def _gen_arrayexpr_getitem(equiv_set, var, parfor_index_tuple_var, all_parfor_indices, el_typ, calltypes, typingctx, typemap, init_block, out_ir):
    """if there is implicit dimension broadcast, generate proper access variable
    for getitem. For example, if indices are (i1,i2,i3) but shape is (c1,0,c3),
    generate a tuple with (i1,0,i3) for access.  Another example: for (i1,i2,i3)
    and (c1,c2) generate (i2,i3).
    """
    loc = var.loc
    index_var = parfor_index_tuple_var
    var_typ = typemap[var.name]
    ndims = typemap[var.name].ndim
    num_indices = len(all_parfor_indices)
    size_vars = equiv_set.get_shape(var) or []
    size_consts = [equiv_set.get_equiv_const(x) for x in size_vars]
    if ndims == 0:
        ravel_var = ir.Var(var.scope, mk_unique_var('$ravel'), loc)
        ravel_typ = types.npytypes.Array(dtype=var_typ.dtype, ndim=1, layout='C')
        typemap[ravel_var.name] = ravel_typ
        stmts = ir_utils.gen_np_call('ravel', numpy.ravel, ravel_var, [var], typingctx, typemap, calltypes)
        init_block.body.extend(stmts)
        var = ravel_var
        const_node = ir.Const(0, var.loc)
        const_var = ir.Var(var.scope, mk_unique_var('$const_ind_0'), loc)
        typemap[const_var.name] = types.uintp
        const_assign = ir.Assign(const_node, const_var, loc)
        out_ir.append(const_assign)
        index_var = const_var
    elif ndims == 1:
        index_var = all_parfor_indices[-1]
    elif any([x is not None for x in size_consts]):
        ind_offset = num_indices - ndims
        tuple_var = ir.Var(var.scope, mk_unique_var('$parfor_index_tuple_var_bcast'), loc)
        typemap[tuple_var.name] = types.containers.UniTuple(types.uintp, ndims)
        const_node = ir.Const(0, var.loc)
        const_var = ir.Var(var.scope, mk_unique_var('$const_ind_0'), loc)
        typemap[const_var.name] = types.uintp
        const_assign = ir.Assign(const_node, const_var, loc)
        out_ir.append(const_assign)
        index_vars = []
        for i in reversed(range(ndims)):
            size_var = size_vars[i]
            size_const = size_consts[i]
            if size_const == 1:
                index_vars.append(const_var)
            else:
                index_vars.append(all_parfor_indices[ind_offset + i])
        index_vars = list(reversed(index_vars))
        tuple_call = ir.Expr.build_tuple(index_vars, loc)
        tuple_assign = ir.Assign(tuple_call, tuple_var, loc)
        out_ir.append(tuple_assign)
        index_var = tuple_var
    ir_expr = ir.Expr.getitem(var, index_var, loc)
    calltypes[ir_expr] = signature(el_typ, typemap[var.name], typemap[index_var.name])
    return ir_expr