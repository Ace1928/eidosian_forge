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
def test_par_parent_mp_spawn_par_child_par_parent(self):
    """
        Explicit use of multiprocessing spawn, this is safe.
        Does this:
        1. Start with OpenMP
        2. Spawn to processes using OpenMP
        3. Join spawns
        4. Run some more OpenMP
        """
    body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'spawn\')\n            q = ctx.Queue()\n\n            # Start OpenMP runtime and run on parent via parallel function\n            Z = busy_func(X, Y, q)\n            procs = []\n            for x in range(20): # start a lot to try and get overlap\n                ## fork() + exec() to run some OpenMP on children\n                proc = ctx.Process(target = busy_func, args=(X, Y, q))\n                procs.append(proc)\n                sys.stdout.flush()\n                sys.stderr.flush()\n                proc.start()\n\n            [p.join() for p in procs]\n\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n\n            # Run some more OpenMP on parent\n            Z = busy_func(X, Y, q)\n        '
    runme = self.template % body
    cmdline = [sys.executable, '-c', runme]
    out, err = self.run_cmd(cmdline)
    if self._DEBUG:
        print(out, err)