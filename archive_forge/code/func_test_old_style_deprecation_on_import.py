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
def test_old_style_deprecation_on_import(self):
    from numba.core.config import _old_style_deprecation_msg
    code = '\n        import numba\n        '
    env = {'NUMBA_CAPTURED_ERRORS': 'old_style'}
    _out, err = self._run_in_separate_process(code, env)
    self.assertIn(_old_style_deprecation_msg, err.decode())
    env = {'NUMBA_CAPTURED_ERRORS': ''}
    _out, err = self._run_in_separate_process(code, env)
    self.assertNotIn('NumbaPendingDeprecationWarning', err.decode())
    env = {'NUMBA_CAPTURED_ERRORS': 'new_style'}
    _out, err = self._run_in_separate_process(code, env)
    self.assertNotIn('NumbaPendingDeprecationWarning', err.decode())