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
class BaseNativeLowering(abc.ABC, LoweringPass):
    """The base class for a lowering pass. The lowering functionality must be
    specified in inheriting classes by providing an appropriate lowering class
    implementation in the overridden `lowering_class` property."""
    _name = None

    def __init__(self):
        LoweringPass.__init__(self)

    @property
    @abc.abstractmethod
    def lowering_class(self):
        """Returns the class that performs the lowering of the IR describing the
        function that is the target of the current compilation."""
        pass

    def run_pass(self, state):
        if state.library is None:
            codegen = state.targetctx.codegen()
            state.library = codegen.create_library(state.func_id.func_qualname)
            state.library.enable_object_caching()
        library = state.library
        targetctx = state.targetctx
        interp = state.func_ir
        typemap = state.typemap
        restype = state.return_type
        calltypes = state.calltypes
        flags = state.flags
        metadata = state.metadata
        pre_stats = llvm.passmanagers.dump_refprune_stats()
        msg = 'Function %s failed at nopython mode lowering' % (state.func_id.func_name,)
        with fallback_context(state, msg):
            fndesc = funcdesc.PythonFunctionDescriptor.from_specialized_function(interp, typemap, restype, calltypes, mangler=targetctx.mangler, inline=flags.forceinline, noalias=flags.noalias, abi_tags=[flags.get_mangle_string()])
            with targetctx.push_code_library(library):
                lower = self.lowering_class(targetctx, library, fndesc, interp, metadata=metadata)
                lower.lower()
                if not flags.no_cpython_wrapper:
                    lower.create_cpython_wrapper(flags.release_gil)
                if not flags.no_cfunc_wrapper:
                    for t in state.args:
                        if isinstance(t, (types.Omitted, types.Generator)):
                            break
                    else:
                        if isinstance(restype, (types.Optional, types.Generator)):
                            pass
                        else:
                            lower.create_cfunc_wrapper()
                env = lower.env
                call_helper = lower.call_helper
                del lower
            from numba.core.compiler import _LowerResult
            if flags.no_compile:
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=None, env=env)
            else:
                cfunc = targetctx.get_executable(library, fndesc, env)
                targetctx.insert_user_function(cfunc, fndesc, [library])
                state['cr'] = _LowerResult(fndesc, call_helper, cfunc=cfunc, env=env)
            post_stats = llvm.passmanagers.dump_refprune_stats()
            metadata['prune_stats'] = post_stats - pre_stats
            metadata['llvm_pass_timings'] = library.recorded_timings
        return True