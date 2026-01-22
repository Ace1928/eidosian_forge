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
class NopythonRewrites(FunctionPass):
    _name = 'nopython_rewrites'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        """
        Perform any intermediate representation rewrites after type
        inference.
        """
        assert state.func_ir
        assert isinstance(getattr(state, 'typemap', None), dict)
        assert isinstance(getattr(state, 'calltypes', None), dict)
        msg = 'Internal error in post-inference rewriting pass encountered during compilation of function "%s"' % (state.func_id.func_name,)
        pp = postproc.PostProcessor(state.func_ir)
        pp.run(True)
        with fallback_context(state, msg):
            rewrites.rewrite_registry.apply('after-inference', state)
        pp.remove_dels()
        return True