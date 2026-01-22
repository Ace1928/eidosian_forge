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
def _numpy_map_to_parfor(self, equiv_set, call_name, lhs, args, kws, expr):
    """generate parfor from Numpy calls that are maps.
        """
    pass_states = self.pass_states
    scope = lhs.scope
    loc = lhs.loc
    arr_typ = pass_states.typemap[lhs.name]
    el_typ = arr_typ.dtype
    size_vars = equiv_set.get_shape(lhs)
    if size_vars is None:
        if config.DEBUG_ARRAY_OPT >= 1:
            print('Could not convert numpy map to parfor, unknown size')
        return None
    index_vars, loopnests = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
    init_block = ir.Block(scope, loc)
    init_block.body = mk_alloc(pass_states.typingctx, pass_states.typemap, pass_states.calltypes, lhs, tuple(size_vars), el_typ, scope, loc, pass_states.typemap[lhs.name])
    body_label = next_label()
    body_block = ir.Block(scope, loc)
    expr_out_var = ir.Var(scope, mk_unique_var('$expr_out_var'), loc)
    pass_states.typemap[expr_out_var.name] = el_typ
    index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block)
    if call_name == 'zeros':
        value = ir.Const(el_typ(0), loc)
    elif call_name == 'ones':
        value = ir.Const(el_typ(1), loc)
    elif call_name in random_calls:
        _remove_size_arg(call_name, expr)
        new_arg_typs, new_kw_types = _get_call_arg_types(expr, pass_states.typemap)
        pass_states.calltypes.pop(expr)
        pass_states.calltypes[expr] = pass_states.typemap[expr.func.name].get_call_type(typing.Context(), new_arg_typs, new_kw_types)
        value = expr
    else:
        raise NotImplementedError('Map of numpy.{} to parfor is not implemented'.format(call_name))
    value_assign = ir.Assign(value, expr_out_var, loc)
    body_block.body.append(value_assign)
    setitem_node = ir.SetItem(lhs, index_var, expr_out_var, loc)
    pass_states.calltypes[setitem_node] = signature(types.none, pass_states.typemap[lhs.name], index_var_typ, el_typ)
    body_block.body.append(setitem_node)
    parfor = Parfor(loopnests, init_block, {}, loc, index_var, equiv_set, ('{} function'.format(call_name), 'NumPy mapping'), pass_states.flags)
    parfor.loop_body = {body_label: body_block}
    if config.DEBUG_ARRAY_OPT >= 1:
        print('generated parfor for numpy map:')
        parfor.dump()
    return parfor