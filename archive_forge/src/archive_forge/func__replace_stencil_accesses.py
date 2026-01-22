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
def _replace_stencil_accesses(self, stencil_ir, parfor_vars, in_args, index_offsets, stencil_func, arg_to_arr_dict):
    """ Convert relative indexing in the stencil kernel to standard indexing
            by adding the loop index variables to the corresponding dimensions
            of the array index tuples.
        """
    stencil_blocks = stencil_ir.blocks
    in_arr = in_args[0]
    in_arg_names = [x.name for x in in_args]
    if 'standard_indexing' in stencil_func.options:
        for x in stencil_func.options['standard_indexing']:
            if x not in arg_to_arr_dict:
                raise ValueError('Standard indexing requested for an array name not present in the stencil kernel definition.')
        standard_indexed = [arg_to_arr_dict[x] for x in stencil_func.options['standard_indexing']]
    else:
        standard_indexed = []
    if in_arr.name in standard_indexed:
        raise ValueError('The first argument to a stencil kernel must use relative indexing, not standard indexing.')
    ndims = self.typemap[in_arr.name].ndim
    scope = in_arr.scope
    loc = in_arr.loc
    need_to_calc_kernel = stencil_func.neighborhood is None
    if need_to_calc_kernel:
        start_lengths = ndims * [0]
        end_lengths = ndims * [0]
    else:
        start_lengths = [x[0] for x in stencil_func.neighborhood]
        end_lengths = [x[1] for x in stencil_func.neighborhood]
    tuple_table = ir_utils.get_tuple_table(stencil_blocks)
    found_relative_index = False
    for label, block in stencil_blocks.items():
        new_body = []
        for stmt in block.body:
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['setitem', 'static_setitem']) and (stmt.value.value.name in in_arg_names) or ((isinstance(stmt, ir.SetItem) or isinstance(stmt, ir.StaticSetItem)) and stmt.target.name in in_arg_names):
                raise ValueError('Assignments to arrays passed to stencil kernels is not allowed.')
            if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op in ['static_getitem', 'getitem']) and (stmt.value.value.name in in_arg_names) and (stmt.value.value.name not in standard_indexed):
                index_list = stmt.value.index
                if ndims == 1:
                    index_list = [index_list]
                elif hasattr(index_list, 'name') and index_list.name in tuple_table:
                    index_list = tuple_table[index_list.name]
                stencil_ir._definitions = ir_utils.build_definitions(stencil_blocks)
                index_list = [_get_const_index_expr(stencil_ir, self.func_ir, v) for v in index_list]
                if index_offsets:
                    index_list = self._add_index_offsets(index_list, list(index_offsets), new_body, scope, loc)
                if need_to_calc_kernel:
                    if isinstance(index_list, ir.Var) or any([not isinstance(v, int) for v in index_list]):
                        raise ValueError('Variable stencil index only possible with known neighborhood')
                    start_lengths = list(map(min, start_lengths, index_list))
                    end_lengths = list(map(max, end_lengths, index_list))
                    found_relative_index = True
                index_vars = self._add_index_offsets(parfor_vars, list(index_list), new_body, scope, loc)
                if ndims == 1:
                    ind_var = index_vars[0]
                else:
                    ind_var = ir.Var(scope, mk_unique_var('$parfor_index_ind_var'), loc)
                    self.typemap[ind_var.name] = types.containers.UniTuple(types.intp, ndims)
                    tuple_call = ir.Expr.build_tuple(index_vars, loc)
                    tuple_assign = ir.Assign(tuple_call, ind_var, loc)
                    new_body.append(tuple_assign)
                if all([self.typemap[v.name] == types.intp for v in index_vars]):
                    getitem_return_typ = self.typemap[stmt.value.value.name].dtype
                else:
                    getitem_return_typ = self.typemap[stmt.value.value.name]
                getitem_call = ir.Expr.getitem(stmt.value.value, ind_var, loc)
                self.calltypes[getitem_call] = signature(getitem_return_typ, self.typemap[stmt.value.value.name], self.typemap[ind_var.name])
                stmt.value = getitem_call
            new_body.append(stmt)
        block.body = new_body
    if need_to_calc_kernel and (not found_relative_index):
        raise ValueError('Stencil kernel with no accesses to relatively indexed arrays.')
    return (start_lengths, end_lengths)