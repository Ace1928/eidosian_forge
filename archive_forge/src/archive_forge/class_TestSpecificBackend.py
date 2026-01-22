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
class TestSpecificBackend(TestInSubprocess, TestParallelBackendBase):
    """
    This is quite contrived, for each test in the TestParallelBackend tests it
    generates a test that will run the TestParallelBackend test in a new python
    process with an environment modified to ensure a specific threadsafe backend
    is used. This is with view of testing the backends independently and in an
    isolated manner such that if they hang/crash/have issues, it doesn't kill
    the test suite.
    """
    _DEBUG = False

    @classmethod
    def _inject(cls, p, name, backend, backend_guard):
        themod = cls.__module__
        thecls = TestParallelBackend.__name__
        methname = 'test_' + p + '_' + name
        injected_method = '%s.%s.%s' % (themod, thecls, methname)

        def test_template(self):
            o, e = self.run_test_in_separate_process(injected_method, backend)
            if self._DEBUG:
                print('stdout:\n "%s"\n stderr:\n "%s"' % (o, e))
            m = re.search("\\.\\.\\. skipped '(.*?)'", e)
            if m is not None:
                self.skipTest(m.group(1))
            self.assertIn('OK', e)
            self.assertTrue('FAIL' not in e)
            self.assertTrue('ERROR' not in e)
        injected_test = 'test_%s_%s_%s' % (p, name, backend)
        setattr(cls, injected_test, tag('long_running')(backend_guard(test_template)))

    @classmethod
    def generate(cls):
        for backend, backend_guard in cls.backends.items():
            for p in cls.parallelism:
                for name in cls.runners.keys():
                    if p in ('multiprocessing_fork', 'random') and backend == 'omp' and sys.platform.startswith('linux'):
                        continue
                    if p in ('threading', 'random') and backend == 'workqueue':
                        continue
                    cls._inject(p, name, backend, backend_guard)