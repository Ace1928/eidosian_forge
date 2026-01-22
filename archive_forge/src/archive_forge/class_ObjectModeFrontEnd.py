from numba.core import (types, typing, funcdesc, config, pylowering, transforms,
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from collections import defaultdict
import warnings
@register_pass(mutates_CFG=True, analysis_only=False)
class ObjectModeFrontEnd(FunctionPass):
    _name = 'object_mode_front_end'

    def __init__(self):
        FunctionPass.__init__(self)

    def _frontend_looplift(self, state):
        """
        Loop lifting analysis and transformation
        """
        loop_flags = state.flags.copy()
        outer_flags = state.flags.copy()
        outer_flags.enable_looplift = False
        loop_flags.enable_looplift = False
        if not state.flags.enable_pyobject_looplift:
            loop_flags.enable_pyobject = False
        loop_flags.enable_ssa = False
        main, loops = transforms.loop_lifting(state.func_ir, typingctx=state.typingctx, targetctx=state.targetctx, locals=state.locals, flags=loop_flags)
        if loops:
            if config.DEBUG_FRONTEND or config.DEBUG:
                for loop in loops:
                    print('Lifting loop', loop.get_source_location())
            from numba.core.compiler import compile_ir
            cres = compile_ir(state.typingctx, state.targetctx, main, state.args, state.return_type, outer_flags, state.locals, lifted=tuple(loops), lifted_from=None, is_lifted_loop=True)
            return cres

    def run_pass(self, state):
        from numba.core.compiler import _EarlyPipelineCompletion
        if state.flags.enable_looplift:
            assert not state.lifted
            cres = self._frontend_looplift(state)
            if cres is not None:
                raise _EarlyPipelineCompletion(cres)
        state.typemap = defaultdict(lambda: types.pyobject)
        state.calltypes = defaultdict(lambda: types.pyobject)
        state.return_type = types.pyobject
        return True