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
class TestThreadingLayerSelection(ThreadLayerTestHelper):
    """
    Checks that numba.threading_layer() reports correctly.
    """
    _DEBUG = False
    backends = {'tbb': skip_no_tbb, 'omp': skip_no_omp, 'workqueue': unittest.skipIf(False, '')}

    @classmethod
    def _inject(cls, backend, backend_guard):

        def test_template(self):
            body = "if 1:\n                X = np.arange(1000000.)\n                Y = np.arange(1000000.)\n                Z = busy_func(X, Y)\n                assert numba.threading_layer() == '%s'\n            "
            runme = self.template % (body % backend)
            cmdline = [sys.executable, '-c', runme]
            env = os.environ.copy()
            env['NUMBA_THREADING_LAYER'] = str(backend)
            out, err = self.run_cmd(cmdline, env=env)
            if self._DEBUG:
                print(out, err)
        injected_test = 'test_threading_layer_selector_%s' % backend
        setattr(cls, injected_test, tag('important')(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        for backend, backend_guard in cls.backends.items():
            cls._inject(backend, backend_guard)