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
class RewriteSemanticConstants(FunctionPass):
    _name = 'rewrite_semantic_constants'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        This prunes dead branches, a dead branch is one which is derivable as
        not taken at compile time purely based on const/literal evaluation.
        """
        assert state.func_ir
        msg = 'Internal error in pre-inference dead branch pruning pass encountered during compilation of function "%s"' % (state.func_id.func_name,)
        with fallback_context(state, msg):
            rewrite_semantic_constants(state.func_ir, state.args)
        return True