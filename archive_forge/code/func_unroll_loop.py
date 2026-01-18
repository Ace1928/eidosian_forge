from collections import defaultdict, namedtuple
from contextlib import contextmanager
from copy import deepcopy, copy
import warnings
from numba.core.compiler_machinery import (FunctionPass, AnalysisPass,
from numba.core import (errors, types, ir, bytecode, postproc, rewrites, config,
from numba.misc.special import literal_unroll
from numba.core.analysis import (dead_branch_prune, rewrite_semantic_constants,
from numba.core.ir_utils import (guard, resolve_func_from_module, simplify_CFG,
from numba.core.ssa import reconstruct_ssa
from numba.core import interpreter
def unroll_loop(self, state, loop_info):
    func_ir = state.func_ir
    getitem_target = loop_info.arg
    target_ty = state.typemap[getitem_target.name]
    assert isinstance(target_ty, self._accepted_types)
    tuple_getitem = []
    for lbl in loop_info.loop.body:
        blk = func_ir.blocks[lbl]
        for stmt in blk.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Expr) and stmt.value.op == 'getitem':
                    if stmt.value.value != getitem_target:
                        dfn = func_ir.get_definition(stmt.value.value)
                        try:
                            args = getattr(dfn, 'args', False)
                        except KeyError:
                            continue
                        if not args:
                            continue
                        if not args[0] == getitem_target:
                            continue
                    target_ty = state.typemap[getitem_target.name]
                    if not isinstance(target_ty, self._accepted_types):
                        continue
                    tuple_getitem.append(stmt)
    if not tuple_getitem:
        msg = "Loop unrolling analysis has failed, there's no getitem in loop body that conforms to literal_unroll requirements."
        LOC = func_ir.blocks[loop_info.loop.header].loc
        raise errors.CompilerError(msg, LOC)
    switch_data = self.analyse_tuple(target_ty)
    index = func_ir._definitions[tuple_getitem[0].value.index.name][0]
    branches = self.gen_switch(switch_data, index)
    for item in tuple_getitem:
        old = item.value
        new = ir.Expr.typed_getitem(old.value, types.void, old.index, old.loc)
        item.value = new
    this_loop = loop_info.loop
    this_loop_body = this_loop.body - set([this_loop.header])
    loop_blocks = {x: func_ir.blocks[x] for x in this_loop_body}
    new_ir = func_ir.derive(loop_blocks)
    usedefs = compute_use_defs(func_ir.blocks)
    idx = this_loop.header
    keep = set()
    keep |= usedefs.usemap[idx] | usedefs.defmap[idx]
    keep |= func_ir.variable_lifetime.livemap[idx]
    dont_replace = [x for x in keep]
    unrolled_body = self.inject_loop_body(branches, new_ir, max(func_ir.blocks.keys()) + 1, dont_replace, switch_data)
    blks = state.func_ir.blocks
    the_scope = next(iter(blks.values())).scope
    orig_lbl = tuple(this_loop_body)
    replace, *delete = orig_lbl
    unroll, header_block = (unrolled_body, this_loop.header)
    unroll_lbl = [x for x in sorted(unroll.blocks.keys())]
    blks[replace] = transfer_scope(unroll.blocks[unroll_lbl[0]], the_scope)
    [blks.pop(d) for d in delete]
    for k in unroll_lbl[1:]:
        blks[k] = transfer_scope(unroll.blocks[k], the_scope)
    blks[header_block].body[-1].truebr = replace