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
@register_pass(mutates_CFG=False, analysis_only=False)
class ReconstructSSA(FunctionPass):
    """Perform SSA-reconstruction

    Produces minimal SSA.
    """
    _name = 'reconstruct_ssa'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.func_ir = reconstruct_ssa(state.func_ir)
        self._patch_locals(state)
        state.func_ir._definitions = build_definitions(state.func_ir.blocks)
        post_proc = postproc.PostProcessor(state.func_ir)
        post_proc.run(emit_dels=False)
        if config.DEBUG or config.DUMP_SSA:
            name = state.func_ir.func_id.func_qualname
            print(f'SSA IR DUMP: {name}'.center(80, '-'))
            state.func_ir.dump()
        return True

    def _patch_locals(self, state):
        locals_dict = state.get('locals')
        if locals_dict is None:
            return
        first_blk, *_ = state.func_ir.blocks.values()
        scope = first_blk.scope
        for parent, redefs in scope.var_redefinitions.items():
            if parent in locals_dict:
                typ = locals_dict[parent]
                for derived in redefs:
                    locals_dict[derived] = typ