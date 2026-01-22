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
class LiteralUnroll(FunctionPass):
    """Implement the literal_unroll semantics"""
    _name = 'literal_unroll'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        found = False
        func_ir = state.func_ir
        for blk in func_ir.blocks.values():
            for asgn in blk.find_insts(ir.Assign):
                if isinstance(asgn.value, (ir.Global, ir.FreeVar)):
                    if asgn.value.value is literal_unroll:
                        found = True
                        break
            if found:
                break
        if not found:
            return False
        from numba.core.compiler_machinery import PassManager
        from numba.core.typed_passes import PartialTypeInference
        pm = PassManager('literal_unroll_subpipeline')
        pm.add_pass(PartialTypeInference, 'performs partial type inference')
        pm.add_pass(TransformLiteralUnrollConstListToTuple, 'switch const list for tuples')
        pm.add_pass(PartialTypeInference, 'performs partial type inference')
        pm.add_pass(IterLoopCanonicalization, 'switch iter loops for range driven loops')
        pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
        pm.add_pass(MixedContainerUnroller, 'performs mixed container unroll')
        pm.add_pass(GenericRewrites, 'Generic Rewrites')
        pm.add_pass(RewriteSemanticConstants, 'rewrite semantic constants')
        pm.finalize()
        pm.run(state)
        return True