from collections import namedtuple
import numpy as np
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba import njit, typed, literal_unroll, prange
from numba.core import types, errors, ir
from numba.testing import unittest
from numba.core.extending import overload
from numba.core.compiler_machinery import (PassManager, register_pass,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (FixupArgs, TranslateByteCode,
from numba.core.typed_passes import (NopythonTypeInference, IRLegalization,
from numba.core.ir_utils import (compute_cfg_from_blocks, flatten_labels)
from numba.core.types.functions import _header_lead
@register_pass(mutates_CFG=False, analysis_only=False)
class ResetTypeInfo(FunctionPass):
    _name = 'reset_the_type_information'

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        state.typemap = None
        state.return_type = None
        state.calltypes = None
        return True