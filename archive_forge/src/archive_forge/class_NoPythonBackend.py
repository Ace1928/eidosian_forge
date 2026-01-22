import abc
from contextlib import contextmanager
from collections import defaultdict, namedtuple
from functools import partial
from copy import copy
import warnings
from numba.core import (errors, types, typing, ir, funcdesc, rewrites,
from numba.parfors.parfor import PreParforPass as _parfor_PreParforPass
from numba.parfors.parfor import ParforPass as _parfor_ParforPass
from numba.parfors.parfor import ParforFusionPass as _parfor_ParforFusionPass
from numba.parfors.parfor import ParforPreLoweringPass as \
from numba.parfors.parfor import Parfor
from numba.parfors.parfor_lowering import ParforLower
from numba.core.compiler_machinery import (FunctionPass, LoweringPass,
from numba.core.annotations import type_annotations
from numba.core.ir_utils import (raise_on_unsupported_feature, warn_deprecated,
from numba.core import postproc
from llvmlite import binding as llvm
@register_pass(mutates_CFG=True, analysis_only=False)
class NoPythonBackend(LoweringPass):
    _name = 'nopython_backend'

    def __init__(self):
        LoweringPass.__init__(self)

    def run_pass(self, state):
        """
        Back-end: Generate LLVM IR from Numba IR, compile to machine code
        """
        lowered = state['cr']
        signature = typing.signature(state.return_type, *state.args)
        from numba.core.compiler import compile_result
        state.cr = compile_result(typing_context=state.typingctx, target_context=state.targetctx, entry_point=lowered.cfunc, typing_error=state.status.fail_reason, type_annotation=state.type_annotation, library=state.library, call_helper=lowered.call_helper, signature=signature, objectmode=False, lifted=state.lifted, fndesc=lowered.fndesc, environment=lowered.env, metadata=state.metadata, reload_init=state.reload_init)
        return True