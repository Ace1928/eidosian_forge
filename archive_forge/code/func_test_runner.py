import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
@unittest.skipUnless(not skipped, 'Not implemented')
def test_runner(self):
    ctx = mp.get_context('spawn')
    q = ctx.Queue()
    p = ctx.Process(target=type(self).mp_runner, args=[testname, q])
    p.start()
    term_or_timeout = p.join(timeout=30)
    exitcode = p.exitcode
    if term_or_timeout is None:
        if exitcode is None:
            self.fail('Process timed out.')
        elif exitcode < 0:
            self.fail(f'Process terminated with signal {-exitcode}.')
    self.assertEqual(exitcode, 0, msg='process ended unexpectedly')
    out = q.get()
    status = out['status']
    msg = out['msg']
    self.assertTrue(status, msg=msg)