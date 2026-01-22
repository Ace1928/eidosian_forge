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
class ConvertLoopPass:
    """Build Parfor nodes from prange loops.
    """

    def __init__(self, pass_states):
        self.pass_states = pass_states
        self.rewritten = []

    def run(self, blocks):
        pass_states = self.pass_states
        call_table, _ = get_call_table(blocks)
        cfg = compute_cfg_from_blocks(blocks)
        usedefs = compute_use_defs(blocks)
        live_map = compute_live_map(cfg, blocks, usedefs.usemap, usedefs.defmap)
        loops = cfg.loops()
        sized_loops = [(loops[k], len(loops[k].body)) for k in loops.keys()]
        moved_blocks = []
        for loop, s in sorted(sized_loops, key=lambda tup: tup[1]):
            if len(loop.entries) != 1 or len(loop.exits) != 1:
                if not config.DISABLE_PERFORMANCE_WARNINGS:
                    for entry in loop.entries:
                        for inst in blocks[entry].body:
                            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_parallel_loop(inst.value.func.name, call_table):
                                msg = '\nprange or pndindex loop will not be executed in parallel due to there being more than one entry to or exit from the loop (e.g., an assertion).'
                                warnings.warn(errors.NumbaPerformanceWarning(msg, inst.loc))
                continue
            entry = list(loop.entries)[0]
            for inst in blocks[entry].body:
                if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_parallel_loop(inst.value.func.name, call_table):
                    body_labels = [l for l in loop.body if l in blocks and l != loop.header]
                    args = inst.value.args
                    loop_kind, loop_replacing = self._get_loop_kind(inst.value.func.name, call_table)
                    header_body = blocks[loop.header].body[:-1]
                    loop_index = None
                    for hbi, stmt in enumerate(header_body):
                        if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Expr) and (stmt.value.op == 'pair_first'):
                            loop_index = stmt.target.name
                            li_index = hbi
                            break
                    assert loop_index is not None
                    header_body = header_body[:li_index] + header_body[li_index + 1:]
                    cps, _ = get_block_copies({0: blocks[loop.header]}, pass_states.typemap)
                    cps = cps[0]
                    loop_index_vars = set((t for t, v in cps if v == loop_index))
                    loop_index_vars.add(loop_index)
                    scope = blocks[entry].scope
                    loc = inst.loc
                    equiv_set = pass_states.array_analysis.get_equiv_set(loop.header)
                    init_block = ir.Block(scope, loc)
                    init_block.body = self._get_prange_init_block(blocks[entry], call_table, args)
                    loop_body = {l: blocks[l] for l in body_labels}
                    end_label = next_label()
                    loop_body[end_label] = ir.Block(scope, loc)
                    bodydefs = set()
                    for bl in body_labels:
                        bodydefs = bodydefs.union(usedefs.defmap[bl])
                    exit_lives = set()
                    for bl in loop.exits:
                        exit_lives = exit_lives.union(live_map[bl])
                    races = bodydefs.intersection(exit_lives)
                    races = races.intersection({x for x in races if not isinstance(pass_states.typemap[x], types.misc.Module)})
                    for l in body_labels:
                        last_inst = loop_body[l].body[-1]
                        if isinstance(last_inst, ir.Jump) and last_inst.target == loop.header:
                            last_inst.target = end_label

                    def find_indexed_arrays():
                        """find expressions that involve getitem using the
                        index variable. Return both the arrays and expressions.
                        """
                        indices = copy.copy(loop_index_vars)
                        for block in loop_body.values():
                            for inst in block.find_insts(ir.Assign):
                                if isinstance(inst.value, ir.Var) and inst.value.name in indices:
                                    indices.add(inst.target.name)
                        arrs = []
                        exprs = []
                        for block in loop_body.values():
                            for inst in block.body:
                                lv = set((x.name for x in inst.list_vars()))
                                if lv & indices:
                                    if lv.issubset(indices):
                                        continue
                                    require(isinstance(inst, ir.Assign))
                                    expr = inst.value
                                    require(isinstance(expr, ir.Expr) and expr.op in ['getitem', 'static_getitem'])
                                    arrs.append(expr.value.name)
                                    exprs.append(expr)
                        return (arrs, exprs)
                    mask_var = None
                    mask_indices = None

                    def find_mask_from_size(size_var):
                        """Find the case where size_var is defined by A[M].shape,
                        where M is a boolean array.
                        """
                        size_def = get_definition(pass_states.func_ir, size_var)
                        require(size_def and isinstance(size_def, ir.Expr) and (size_def.op == 'getattr') and (size_def.attr == 'shape'))
                        arr_var = size_def.value
                        live_vars = set.union(*[live_map[l] for l in loop.exits])
                        index_arrs, index_exprs = find_indexed_arrays()
                        require([arr_var.name] == list(index_arrs))
                        require(arr_var.name not in live_vars)
                        arr_def = get_definition(pass_states.func_ir, size_def.value)
                        result = _find_mask(pass_states.typemap, pass_states.func_ir, arr_def)
                        raise AssertionError('unreachable')
                        for expr in index_exprs:
                            expr.value = result[0]
                        return result
                    unsigned_index = True
                    if loop_kind == 'pndindex':
                        assert equiv_set.has_shape(args[0])
                        result = guard(find_mask_from_size, args[0])
                        if result:
                            in_arr, mask_var, mask_typ, mask_indices = result
                        else:
                            in_arr = args[0]
                        assert isinstance(in_arr, ir.Var)
                        in_arr_typ = pass_states.typemap[in_arr.name]
                        if isinstance(in_arr_typ, types.Integer):
                            index_var = ir.Var(scope, mk_unique_var('parfor_index'), loc)
                            pass_states.typemap[index_var.name] = types.uintp
                            loops = [LoopNest(index_var, 0, in_arr, 1)]
                            index_vars = [index_var]
                        else:
                            size_vars = equiv_set.get_shape(in_arr if mask_indices is None else mask_var)
                            index_vars, loops = _mk_parfor_loops(pass_states.typemap, size_vars, scope, loc)
                        assert len(loops) > 0
                        orig_index = index_vars
                        if mask_indices:
                            index_vars = tuple((x if x else index_vars[0] for x in mask_indices))
                        first_body_block = loop_body[min(loop_body.keys())]
                        body_block = ir.Block(scope, loc)
                        index_var, index_var_typ = _make_index_var(pass_states.typemap, scope, index_vars, body_block, force_tuple=True)
                        body = body_block.body + first_body_block.body
                        first_body_block.body = body
                        if mask_indices:
                            orig_index_var = orig_index[0]
                        else:
                            orig_index_var = index_var
                        if mask_var is not None:
                            raise AssertionError('unreachable')
                            body_label = next_label()
                            loop_body = add_offset_to_labels(loop_body, body_label - min(loop_body.keys()) + 1)
                            labels = loop_body.keys()
                            true_label = min(labels)
                            false_label = max(labels)
                            body_block = ir.Block(scope, loc)
                            loop_body[body_label] = body_block
                            mask = ir.Var(scope, mk_unique_var('$mask_val'), loc)
                            pass_states.typemap[mask.name] = mask_typ
                            mask_val = ir.Expr.getitem(mask_var, orig_index_var, loc)
                            body_block.body.extend([ir.Assign(mask_val, mask, loc), ir.Branch(mask, true_label, false_label, loc)])
                    else:
                        start = 0
                        step = 1
                        size_var = args[0]
                        if len(args) == 2:
                            start = args[0]
                            size_var = args[1]
                        if len(args) == 3:
                            start = args[0]
                            size_var = args[1]
                            try:
                                step = pass_states.func_ir.get_definition(args[2])
                            except KeyError:
                                raise errors.UnsupportedRewriteError('Only known step size is supported for prange', loc=inst.loc)
                            if not isinstance(step, ir.Const):
                                raise errors.UnsupportedRewriteError('Only constant step size is supported for prange', loc=inst.loc)
                            step = step.value
                            if step != 1:
                                raise errors.UnsupportedRewriteError('Only constant step size of 1 is supported for prange', loc=inst.loc)
                        index_var = ir.Var(scope, mk_unique_var('parfor_index'), loc)
                        if isinstance(start, int) and start >= 0:
                            index_var_typ = types.uintp
                        else:
                            index_var_typ = types.intp
                            unsigned_index = False
                        loops = [LoopNest(index_var, start, size_var, step)]
                        pass_states.typemap[index_var.name] = index_var_typ
                        first_body_label = min(loop_body.keys())
                        loop_body[first_body_label].body = header_body + loop_body[first_body_label].body
                    index_var_map = {v: index_var for v in loop_index_vars}
                    replace_vars(loop_body, index_var_map)
                    if unsigned_index:
                        self._replace_loop_access_indices(loop_body, loop_index_vars, index_var)
                    parfor = Parfor(loops, init_block, loop_body, loc, orig_index_var if mask_indices else index_var, equiv_set, ('prange', loop_kind, loop_replacing), pass_states.flags, races=races)
                    blocks[loop.header].body = [parfor]
                    blocks[loop.header].body.extend(header_body)
                    blocks[loop.header].body.append(ir.Jump(list(loop.exits)[0], loc))
                    self.rewritten.append(dict(old_loop=loop, new=parfor, reason='loop'))
                    for l in body_labels:
                        if l != loop.header:
                            blocks.pop(l)
                    if config.DEBUG_ARRAY_OPT >= 1:
                        print('parfor from loop')
                        parfor.dump()

    def _is_parallel_loop(self, func_var, call_table):
        if func_var not in call_table:
            return False
        call = call_table[func_var]
        return len(call) > 0 and (call[0] == 'prange' or call[0] == prange or call[0] == 'internal_prange' or (call[0] == internal_prange) or (call[0] == 'pndindex') or (call[0] == pndindex))

    def _get_loop_kind(self, func_var, call_table):
        """see if prange is user prange or internal"""
        pass_states = self.pass_states
        assert func_var in call_table
        call = call_table[func_var]
        assert len(call) > 0
        kind = ('user', '')
        if call[0] == 'internal_prange' or call[0] == internal_prange:
            try:
                kind = ('internal', (pass_states.swapped_fns[func_var][0], pass_states.swapped_fns[func_var][-1]))
            except KeyError:
                kind = ('internal', ('', ''))
        elif call[0] == 'pndindex' or call[0] == pndindex:
            kind = ('pndindex', '')
        return kind

    def _get_prange_init_block(self, entry_block, call_table, prange_args):
        """
        If there is init_prange, find the code between init_prange and prange
        calls. Remove the code from entry_block and return it.
        """
        init_call_ind = -1
        prange_call_ind = -1
        init_body = []
        for i, inst in enumerate(entry_block.body):
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_prange_init(inst.value.func.name, call_table):
                init_call_ind = i
            if isinstance(inst, ir.Assign) and isinstance(inst.value, ir.Expr) and (inst.value.op == 'call') and self._is_parallel_loop(inst.value.func.name, call_table):
                prange_call_ind = i
        if init_call_ind != -1 and prange_call_ind != -1:
            arg_related_vars = {v.name for v in prange_args}
            saved_nodes = []
            for i in reversed(range(init_call_ind + 1, prange_call_ind)):
                inst = entry_block.body[i]
                inst_vars = {v.name for v in inst.list_vars()}
                if arg_related_vars & inst_vars:
                    arg_related_vars |= inst_vars
                    saved_nodes.append(inst)
                else:
                    init_body.append(inst)
            init_body.reverse()
            saved_nodes.reverse()
            entry_block.body = entry_block.body[:init_call_ind] + saved_nodes + entry_block.body[prange_call_ind + 1:]
        return init_body

    def _is_prange_init(self, func_var, call_table):
        if func_var not in call_table:
            return False
        call = call_table[func_var]
        return len(call) > 0 and (call[0] == 'init_prange' or call[0] == init_prange)

    def _replace_loop_access_indices(self, loop_body, index_set, new_index):
        """
        Replace array access indices in a loop body with a new index.
        index_set has all the variables that are equivalent to loop index.
        """
        index_set.add(new_index.name)
        with dummy_return_in_loop_body(loop_body):
            labels = find_topo_order(loop_body)
        first_label = labels[0]
        added_indices = set()
        for l in labels:
            block = loop_body[l]
            for stmt in block.body:
                if isinstance(stmt, ir.Assign) and isinstance(stmt.value, ir.Var):
                    if l == first_label and stmt.value.name in index_set and (stmt.target.name not in index_set):
                        index_set.add(stmt.target.name)
                        added_indices.add(stmt.target.name)
                    else:
                        scope = block.scope

                        def unver(name):
                            from numba.core import errors
                            try:
                                return scope.get_exact(name).unversioned_name
                            except errors.NotDefinedError:
                                return name
                        if unver(stmt.target.name) in map(unver, index_set) and unver(stmt.target.name) != unver(stmt.value.name):
                            raise errors.UnsupportedRewriteError('Overwrite of parallel loop index', loc=stmt.target.loc)
                if is_get_setitem(stmt):
                    index = index_var_of_get_setitem(stmt)
                    if index is None:
                        continue
                    ind_def = guard(get_definition, self.pass_states.func_ir, index, lhs_only=True)
                    if index.name in index_set or (ind_def is not None and ind_def.name in index_set):
                        set_index_var_of_get_setitem(stmt, new_index)
                    guard(self._replace_multi_dim_ind, ind_def, index_set, new_index)
                if isinstance(stmt, Parfor):
                    self._replace_loop_access_indices(stmt.loop_body, index_set, new_index)
        index_set -= added_indices
        return

    def _replace_multi_dim_ind(self, ind_var, index_set, new_index):
        """
        replace individual indices in multi-dimensional access variable, which
        is a build_tuple
        """
        pass_states = self.pass_states
        require(ind_var is not None)
        require(isinstance(pass_states.typemap[ind_var.name], (types.Tuple, types.UniTuple)))
        ind_def_node = get_definition(pass_states.func_ir, ind_var)
        require(isinstance(ind_def_node, ir.Expr) and ind_def_node.op == 'build_tuple')
        ind_def_node.items = [new_index if v.name in index_set else v for v in ind_def_node.items]