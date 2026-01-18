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
@unittest.skipUnless(_HAVE_OS_FORK, 'Test needs fork(2)')
def test_workqueue_handles_fork_from_non_main_thread(self):
    runme = 'if 1:\n            from numba import njit, prange, threading_layer\n            import numpy as np\n            import multiprocessing\n\n            if __name__ == "__main__":\n                # Need for force fork context (OSX default is "spawn")\n                multiprocessing.set_start_method(\'fork\')\n\n                @njit(parallel=True)\n                def func(x):\n                    return 10. * x\n\n                arr = np.arange(2.)\n\n                # run in single process to start Numba\'s thread pool\n                np.testing.assert_allclose(func(arr), func.py_func(arr))\n\n                # now run in a multiprocessing pool to get a fork from a\n                # non-main thread\n                with multiprocessing.Pool(10) as p:\n                    result = p.map(func, [arr])\n                np.testing.assert_allclose(result,\n                                           func.py_func(np.expand_dims(arr, 0)))\n\n                assert threading_layer() == "workqueue"\n        '
    cmdline = [sys.executable, '-c', runme]
    env = os.environ.copy()
    env['NUMBA_THREADING_LAYER'] = 'workqueue'
    env['NUMBA_NUM_THREADS'] = '4'
    self.run_cmd(cmdline, env=env)