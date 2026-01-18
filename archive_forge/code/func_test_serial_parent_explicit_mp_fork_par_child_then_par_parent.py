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
@linux_only
def test_serial_parent_explicit_mp_fork_par_child_then_par_parent(self):
    """
        Explicit use of multiprocessing 'fork'.
        Does this:
        1. Start with no OpenMP
        2. Fork to processes using OpenMP
        3. Join forks
        4. Run some OpenMP
        """
    body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'fork\')\n            q = ctx.Queue()\n\n            # this is ok\n            procs = []\n            for x in range(10):\n                # fork() underneath with but no OpenMP in parent, this is ok\n                proc = ctx.Process(target = busy_func, args=(X, Y, q))\n                procs.append(proc)\n                proc.start()\n\n            [p.join() for p in procs]\n\n            # and this is still ok as the OpenMP happened in forks\n            Z = busy_func(X, Y, q)\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n        '
    runme = self.template % body
    cmdline = [sys.executable, '-c', runme]
    out, err = self.run_cmd(cmdline)
    if self._DEBUG:
        print(out, err)