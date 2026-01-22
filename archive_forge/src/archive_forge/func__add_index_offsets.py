import numbers
import copy
import types as pytypes
from operator import add
import operator
import numpy as np
import numba.parfors.parfor
from numba.core import types, ir, rewrites, config, ir_utils
from numba.core.typing.templates import infer_global, AbstractTemplate
from numba.core.typing import signature
from numba.core import  utils, typing
from numba.core.ir_utils import (get_call_table, mk_unique_var,
from numba.core.errors import NumbaValueError
from numba.core.utils import OPERATORS_TO_BUILTINS
from numba.np import numpy_support
def _add_index_offsets(self, index_list, index_offsets, new_body, scope, loc):
    """ Does the actual work of adding loop index variables to the
            relative index constants or variables.
        """
    assert len(index_list) == len(index_offsets)
    if all([isinstance(v, int) for v in index_list + index_offsets]):
        return list(map(add, index_list, index_offsets))
    out_nodes = []
    index_vars = []
    for i in range(len(index_list)):
        old_index_var = index_list[i]
        if isinstance(old_index_var, int):
            old_index_var = ir.Var(scope, mk_unique_var('old_index_var'), loc)
            self.typemap[old_index_var.name] = types.intp
            const_assign = ir.Assign(ir.Const(index_list[i], loc), old_index_var, loc)
            out_nodes.append(const_assign)
        offset_var = index_offsets[i]
        if isinstance(offset_var, int):
            offset_var = ir.Var(scope, mk_unique_var('offset_var'), loc)
            self.typemap[offset_var.name] = types.intp
            const_assign = ir.Assign(ir.Const(index_offsets[i], loc), offset_var, loc)
            out_nodes.append(const_assign)
        if isinstance(old_index_var, slice) or isinstance(self.typemap[old_index_var.name], types.misc.SliceType):
            assert self.typemap[offset_var.name] == types.intp
            index_var = self._add_offset_to_slice(old_index_var, offset_var, out_nodes, scope, loc)
            index_vars.append(index_var)
            continue
        if isinstance(offset_var, slice) or isinstance(self.typemap[offset_var.name], types.misc.SliceType):
            assert self.typemap[old_index_var.name] == types.intp
            index_var = self._add_offset_to_slice(offset_var, old_index_var, out_nodes, scope, loc)
            index_vars.append(index_var)
            continue
        index_var = ir.Var(scope, mk_unique_var('offset_stencil_index'), loc)
        self.typemap[index_var.name] = types.intp
        index_call = ir.Expr.binop(operator.add, old_index_var, offset_var, loc)
        self.calltypes[index_call] = self.typingctx.resolve_function_type(operator.add, (types.intp, types.intp), {})
        index_assign = ir.Assign(index_call, index_var, loc)
        out_nodes.append(index_assign)
        index_vars.append(index_var)
    new_body.extend(out_nodes)
    return index_vars