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
@skip_unless_gnu_omp
class TestForkSafetyIssues(ThreadLayerTestHelper):
    """
    Checks Numba's behaviour in various situations involving GNU OpenMP and fork
    """
    _DEBUG = False

    def test_check_threading_layer_is_gnu(self):
        runme = "if 1:\n            from numba.np.ufunc import omppool\n            assert omppool.openmp_vendor == 'GNU'\n            "
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)

    def test_par_parent_os_fork_par_child(self):
        """
        Whilst normally valid, this actually isn't for Numba invariant of OpenMP
        Checks SIGABRT is received.
        """
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            Z = busy_func(X, Y)\n            pid = os.fork()\n            if pid  == 0:\n                Z = busy_func(X, Y)\n            else:\n                os.wait()\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        try:
            out, err = self.run_cmd(cmdline)
        except AssertionError as e:
            self.assertIn('failed with code -6', str(e))

    def test_par_parent_implicit_mp_fork_par_child(self):
        """
        Implicit use of multiprocessing fork context.
        Does this:
        1. Start with OpenMP
        2. Fork to processes using OpenMP (this is invalid)
        3. Joins fork
        4. Check the exception pushed onto the queue that is a result of
           catching SIGTERM coming from the C++ aborting on illegal fork
           pattern for GNU OpenMP
        """
        body = 'if 1:\n            mp = multiprocessing.get_context(\'fork\')\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            q = mp.Queue()\n\n            # Start OpenMP runtime on parent via parallel function\n            Z = busy_func(X, Y, q)\n\n            # fork() underneath with no exec, will abort\n            proc = mp.Process(target = busy_func, args=(X, Y, q))\n            proc.start()\n\n            err = q.get()\n            assert "Caught SIGTERM" in str(err)\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

    @linux_only
    def test_par_parent_explicit_mp_fork_par_child(self):
        """
        Explicit use of multiprocessing fork context.
        Does this:
        1. Start with OpenMP
        2. Fork to processes using OpenMP (this is invalid)
        3. Joins fork
        4. Check the exception pushed onto the queue that is a result of
           catching SIGTERM coming from the C++ aborting on illegal fork
           pattern for GNU OpenMP
        """
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            ctx = multiprocessing.get_context(\'fork\')\n            q = ctx.Queue()\n\n            # Start OpenMP runtime on parent via parallel function\n            Z = busy_func(X, Y, q)\n\n            # fork() underneath with no exec, will abort\n            proc = ctx.Process(target = busy_func, args=(X, Y, q))\n            proc.start()\n            proc.join()\n\n            err = q.get()\n            assert "Caught SIGTERM" in str(err)\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

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

    def test_serial_parent_implicit_mp_fork_par_child_then_par_parent(self):
        """
        Implicit use of multiprocessing (will be fork, but cannot declare that
        in Py2.7 as there's no process launch context).
        Does this:
        1. Start with no OpenMP
        2. Fork to processes using OpenMP
        3. Join forks
        4. Run some OpenMP
        """
        body = 'if 1:\n            X = np.arange(1000000.)\n            Y = np.arange(1000000.)\n            q = multiprocessing.Queue()\n\n            # this is ok\n            procs = []\n            for x in range(10):\n                # fork() underneath with but no OpenMP in parent, this is ok\n                proc = multiprocessing.Process(target = busy_func,\n                                               args=(X, Y, q))\n                procs.append(proc)\n                proc.start()\n\n            [p.join() for p in procs]\n\n            # and this is still ok as the OpenMP happened in forks\n            Z = busy_func(X, Y, q)\n            try:\n                q.get(False)\n            except multiprocessing.queues.Empty:\n                pass\n            else:\n                raise RuntimeError("Queue was not empty")\n        '
        runme = self.template % body
        cmdline = [sys.executable, '-c', runme]
        out, err = self.run_cmd(cmdline)
        if self._DEBUG:
            print(out, err)

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