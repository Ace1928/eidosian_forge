import sys
import subprocess
import numpy as np
import os
import warnings
from numba import jit, njit, types
from numba.core import errors
from numba.experimental import structref
from numba.extending import (overload, intrinsic, overload_method,
from numba.core.compiler import CompilerBase
from numba.core.untyped_passes import (TranslateByteCode, FixupArgs,
from numba.core.typed_passes import (NopythonTypeInference, DeadCodeElimination,
from numba.core.compiler_machinery import PassManager
from numba.core.types.functions import _err_reasons as error_reasons
from numba.tests.support import (skip_parfors_unsupported, override_config,
import unittest
def _test_old_style_deprecation(self):
    warnings.simplefilter('always', errors.NumbaPendingDeprecationWarning)

    def bar(x):
        pass

    @overload(bar)
    def ol_bar(x):
        raise AttributeError('Invalid attribute')
    with self.assertWarns(errors.NumbaPendingDeprecationWarning) as warns:
        with self.assertRaises(errors.TypingError):

            @njit('void(int64)')
            def foo(x):
                bar(x)
        self.assertIn("Code using Numba extension API maybe depending on 'old_style' error-capturing", str(warns.warnings[0].message))