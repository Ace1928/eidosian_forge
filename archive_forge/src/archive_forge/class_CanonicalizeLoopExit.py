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
@register_pass(mutates_CFG=True, analysis_only=False)
class CanonicalizeLoopExit(FunctionPass):
    """A pass to canonicalize loop exit by splitting it from function exit.
    """
    _name = 'canonicalize_loop_exit'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        fir = state.func_ir
        cfg = compute_cfg_from_blocks(fir.blocks)
        status = False
        for loop in cfg.loops().values():
            for exit_label in loop.exits:
                if exit_label in cfg.exit_points():
                    self._split_exit_block(fir, cfg, exit_label)
                    status = True
        fir._reset_analysis_variables()
        vlt = postproc.VariableLifetime(fir.blocks)
        fir.variable_lifetime = vlt
        return status

    def _split_exit_block(self, fir, cfg, exit_label):
        curblock = fir.blocks[exit_label]
        newlabel = exit_label + 1
        newlabel = find_max_label(fir.blocks) + 1
        fir.blocks[newlabel] = curblock
        newblock = ir.Block(scope=curblock.scope, loc=curblock.loc)
        newblock.append(ir.Jump(newlabel, loc=curblock.loc))
        fir.blocks[exit_label] = newblock
        fir.blocks = rename_labels(fir.blocks)