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
def test_workqueue_aborts_on_nested_parallelism(self):
    """
        Tests workqueue raises sigabrt if a nested parallel call is performed
        """
    runme = 'if 1:\n            from numba import njit, prange\n            import numpy as np\n\n            @njit(parallel=True)\n            def nested(x):\n                for i in prange(len(x)):\n                    x[i] += 1\n\n\n            @njit(parallel=True)\n            def main():\n                Z = np.zeros((5, 10))\n                for i in prange(Z.shape[0]):\n                    nested(Z[i])\n                return Z\n\n            main()\n        '
    cmdline = [sys.executable, '-c', runme]
    env = os.environ.copy()
    env['NUMBA_THREADING_LAYER'] = 'workqueue'
    env['NUMBA_NUM_THREADS'] = '4'
    try:
        out, err = self.run_cmd(cmdline, env=env)
    except AssertionError as e:
        if self._DEBUG:
            print(out, err)
        e_msg = str(e)
        self.assertIn('failed with code', e_msg)
        expected = 'Numba workqueue threading layer is terminating: Concurrent access has been detected.'
        self.assertIn(expected, e_msg)