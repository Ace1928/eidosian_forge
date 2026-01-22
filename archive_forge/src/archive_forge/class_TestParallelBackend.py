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
@unittest.skipUnless(_specific_backends, 'Threading layer not explicit')
class TestParallelBackend(TestParallelBackendBase):
    """ These are like the numba.tests.test_threadsafety tests but designed
    instead to torture the parallel backend.
    If a suitable backend is supplied via NUMBA_THREADING_LAYER these tests
    can be run directly. This test class cannot be run using the multiprocessing
    option to the test runner (i.e. `./runtests -m`) as daemon processes cannot
    have children.
    """

    @classmethod
    def generate(cls):
        for p in cls.parallelism:
            for name, impl in cls.runners.items():
                methname = 'test_' + p + '_' + name

                def methgen(impl, p):

                    def test_method(self):
                        selfproc = multiprocessing.current_process()
                        if selfproc.daemon:
                            _msg = 'daemonized processes cannot have children'
                            self.skipTest(_msg)
                        else:
                            self.run_compile(impl, parallelism=p)
                    return test_method
                fn = methgen(impl, p)
                fn.__name__ = methname
                setattr(cls, methname, fn)