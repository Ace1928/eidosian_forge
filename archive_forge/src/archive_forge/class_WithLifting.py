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
class WithLifting(FunctionPass):
    _name = 'with_lifting'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Extract with-contexts
        """
        main, withs = transforms.with_lifting(func_ir=state.func_ir, typingctx=state.typingctx, targetctx=state.targetctx, flags=state.flags, locals=state.locals)
        if withs:
            from numba.core.compiler import compile_ir, _EarlyPipelineCompletion
            cres = compile_ir(state.typingctx, state.targetctx, main, state.args, state.return_type, state.flags, state.locals, lifted=tuple(withs), lifted_from=None, pipeline_class=type(state.pipeline))
            raise _EarlyPipelineCompletion(cres)
        return True