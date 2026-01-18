import faulthandler
import itertools
import multiprocessing
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
import unittest
import numpy as np
from numba import jit, vectorize, guvectorize, set_num_threads
from numba.tests.support import (temp_directory, override_config, TestCase, tag,
import queue as t_queue
from numba.testing.main import _TIMEOUT as _RUNNER_TIMEOUT
from numba.core import config
@skip_no_omp
def test_omp_stack_overflow(self):
    """
        Tests that OMP does not overflow stack
        """
    runme = 'if 1:\n            from numba import vectorize, threading_layer\n            import numpy as np\n\n            @vectorize([\'f4(f4,f4,f4,f4,f4,f4,f4,f4)\'], target=\'parallel\')\n            def foo(a, b, c, d, e, f, g, h):\n                return a+b+c+d+e+f+g+h\n\n            x = np.ones(2**20, np.float32)\n            foo(*([x]*8))\n            assert threading_layer() == "omp", "omp not found"\n        '
    cmdline = [sys.executable, '-c', runme]
    env = os.environ.copy()
    env['NUMBA_THREADING_LAYER'] = 'omp'
    env['OMP_STACKSIZE'] = '100K'
    self.run_cmd(cmdline, env=env)