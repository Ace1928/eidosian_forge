from collections import namedtuple, defaultdict
import logging
import operator
from numba.core.analysis import compute_cfg_from_blocks, find_top_level_loops
from numba.core import errors, ir, ir_utils
from numba.core.analysis import compute_use_defs, compute_cfg_from_blocks
from numba.core.utils import PYVERSION
def _fix_multi_exit_blocks(func_ir, exit_nodes, *, split_condition=None):
    """Modify the FunctionIR to create a single common exit node given the
    original exit nodes.

    Parameters
    ----------
    func_ir :
        The FunctionIR. Mutated inplace.
    exit_nodes :
        The original exit nodes. A sequence of block keys.
    split_condition : callable or None
        If not None, it is a callable with the signature
        `split_condition(statement)` that determines if the `statement` is the
        splitting point (e.g. `POP_BLOCK`) in an exit node.
        If it's None, the exit node is not split.
    """
    blocks = func_ir.blocks
    any_blk = min(func_ir.blocks.values())
    scope = any_blk.scope
    max_label = max(func_ir.blocks) + 1
    common_block = ir.Block(any_blk.scope, loc=ir.unknown_loc)
    common_label = max_label
    max_label += 1
    blocks[common_label] = common_block
    post_block = ir.Block(any_blk.scope, loc=ir.unknown_loc)
    post_label = max_label
    max_label += 1
    blocks[post_label] = post_block
    remainings = []
    for i, k in enumerate(exit_nodes):
        blk = blocks[k]
        if split_condition is not None:
            for pt, stmt in enumerate(blk.body):
                if split_condition(stmt):
                    break
        else:
            pt = -1
        before = blk.body[:pt]
        after = blk.body[pt:]
        remainings.append(after)
        blk.body = before
        loc = blk.loc
        blk.body.append(ir.Assign(value=ir.Const(i, loc=loc), target=scope.get_or_define('$cp', loc=loc), loc=loc))
        assert not blk.is_terminated
        blk.body.append(ir.Jump(common_label, loc=ir.unknown_loc))
    if split_condition is not None:
        common_block.body.append(remainings[0][0])
    assert not common_block.is_terminated
    common_block.body.append(ir.Jump(post_label, loc=loc))
    remain_blocks = []
    for remain in remainings:
        remain_blocks.append(max_label)
        max_label += 1
    switch_block = post_block
    loc = ir.unknown_loc
    for i, remain in enumerate(remainings):
        match_expr = scope.redefine('$cp_check', loc=loc)
        match_rhs = scope.redefine('$cp_rhs', loc=loc)
        switch_block.body.append(ir.Assign(value=ir.Const(i, loc=loc), target=match_rhs, loc=loc))
        switch_block.body.append(ir.Assign(value=ir.Expr.binop(fn=operator.eq, lhs=scope.get('$cp'), rhs=match_rhs, loc=loc), target=match_expr, loc=loc))
        [jump_target] = remain[-1].get_targets()
        switch_block.body.append(ir.Branch(match_expr, jump_target, remain_blocks[i], loc=loc))
        switch_block = ir.Block(scope=scope, loc=loc)
        blocks[remain_blocks[i]] = switch_block
    switch_block.body.append(ir.Jump(jump_target, loc=loc))
    return (func_ir, common_label)