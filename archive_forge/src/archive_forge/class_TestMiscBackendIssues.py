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
@skip_parfors_unsupported
class TestMiscBackendIssues(ThreadLayerTestHelper):
    """
    Checks fixes for the issues with threading backends implementation
    """
    _DEBUG = False

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

    @skip_no_tbb
    def test_single_thread_tbb(self):
        """
        Tests that TBB works well with single thread
        https://github.com/numba/numba/issues/3440
        """
        runme = 'if 1:\n            from numba import njit, prange, threading_layer\n\n            @njit(parallel=True)\n            def foo(n):\n                acc = 0\n                for i in prange(n):\n                    acc += i\n                return acc\n\n            foo(100)\n            assert threading_layer() == "tbb", "tbb not found"\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'tbb'
        env['NUMBA_NUM_THREADS'] = '1'
        self.run_cmd(cmdline, env=env)

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

    @unittest.skipUnless(_HAVE_OS_FORK, 'Test needs fork(2)')
    def test_workqueue_handles_fork_from_non_main_thread(self):
        runme = 'if 1:\n            from numba import njit, prange, threading_layer\n            import numpy as np\n            import multiprocessing\n\n            if __name__ == "__main__":\n                # Need for force fork context (OSX default is "spawn")\n                multiprocessing.set_start_method(\'fork\')\n\n                @njit(parallel=True)\n                def func(x):\n                    return 10. * x\n\n                arr = np.arange(2.)\n\n                # run in single process to start Numba\'s thread pool\n                np.testing.assert_allclose(func(arr), func.py_func(arr))\n\n                # now run in a multiprocessing pool to get a fork from a\n                # non-main thread\n                with multiprocessing.Pool(10) as p:\n                    result = p.map(func, [arr])\n                np.testing.assert_allclose(result,\n                                           func.py_func(np.expand_dims(arr, 0)))\n\n                assert threading_layer() == "workqueue"\n        '
        cmdline = [sys.executable, '-c', runme]
        env = os.environ.copy()
        env['NUMBA_THREADING_LAYER'] = 'workqueue'
        env['NUMBA_NUM_THREADS'] = '4'
        self.run_cmd(cmdline, env=env)