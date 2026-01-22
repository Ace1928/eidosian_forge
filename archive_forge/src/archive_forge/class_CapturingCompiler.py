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
class CapturingCompiler(CompilerBase):
    """ Simple pipeline that wraps passes with the ResultCapturer pass"""

    def define_pipelines(self):
        pm = PassManager('Capturing Compiler')

        def add_pass(x, y):
            return pm.add_pass(capture(x), y)
        add_pass(TranslateByteCode, 'analyzing bytecode')
        add_pass(FixupArgs, 'fix up args')
        add_pass(IRProcessing, 'processing IR')
        add_pass(LiteralUnroll, 'handles literal_unroll')
        add_pass(NopythonTypeInference, 'nopython frontend')
        add_pass(IRLegalization, 'ensure IR is legal prior to lowering')
        add_pass(NativeLowering, 'native lowering')
        add_pass(NoPythonBackend, 'nopython mode backend')
        pm.finalize()
        return [pm]