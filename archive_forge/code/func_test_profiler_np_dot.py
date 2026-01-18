import cProfile as profiler
import os
import pstats
import subprocess
import sys
import numpy as np
from numba import jit
from numba.tests.support import needs_blas, expected_failure_py312
import unittest
@needs_blas
def test_profiler_np_dot(self):
    code = 'if 1:\n            import cProfile as profiler\n\n            import numpy as np\n\n            from numba import jit\n            from numba.tests.test_profiler import np_dot\n\n            cfunc = jit(nopython=True)(np_dot)\n\n            a = np.arange(16, dtype=np.float32)\n            b = np.arange(16, dtype=np.float32)\n\n            p = profiler.Profile()\n            p.enable()\n            cfunc(a, b)\n            cfunc(a, b)\n            p.disable()\n            '
    subprocess.check_call([sys.executable, '-c', code])