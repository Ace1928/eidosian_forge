import unittest
import unittest.mock
import queue as pyqueue
import textwrap
import time
import io
import itertools
import sys
import os
import gc
import errno
import signal
import array
import socket
import random
import logging
import subprocess
import struct
import operator
import pickle #XXX: use dill?
import weakref
import warnings
import test.support
import test.support.script_helper
from test import support
from test.support import hashlib_helper
from test.support import import_helper
from test.support import os_helper
from test.support import socket_helper
from test.support import threading_helper
from test.support import warnings_helper
import_helper.import_module('multiprocess.synchronize')
import threading
import multiprocess as multiprocessing
import multiprocess.connection
import multiprocess.dummy
import multiprocess.heap
import multiprocess.managers
import multiprocess.pool
import multiprocess.queues
from multiprocess import util
from multiprocess.connection import wait
from multiprocess.managers import BaseManager, BaseProxy, RemoteError
class _TestCondition(BaseTestCase):

    @classmethod
    def f(cls, cond, sleeping, woken, timeout=None):
        cond.acquire()
        sleeping.release()
        cond.wait(timeout)
        woken.release()
        cond.release()

    def assertReachesEventually(self, func, value):
        for i in range(10):
            try:
                if func() == value:
                    break
            except NotImplementedError:
                break
            time.sleep(DELTA)
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(value, func)

    def check_invariant(self, cond):
        if self.TYPE == 'processes':
            try:
                sleepers = cond._sleeping_count.get_value() - cond._woken_count.get_value()
                self.assertEqual(sleepers, 0)
                self.assertEqual(cond._wait_semaphore.get_value(), 0)
            except NotImplementedError:
                pass

    def test_notify(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        p = self.Process(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)
        p = threading.Thread(target=self.f, args=(cond, sleeping, woken))
        p.daemon = True
        p.start()
        self.addCleanup(p.join)
        sleeping.acquire()
        sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify()
        cond.release()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(1, get_value, woken)
        cond.acquire()
        cond.notify()
        cond.release()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(2, get_value, woken)
        self.check_invariant(cond)
        p.join()

    def test_notify_all(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken, TIMEOUT1))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken, TIMEOUT1))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        for i in range(6):
            woken.acquire()
        self.assertReturnsIfImplemented(0, get_value, woken)
        self.check_invariant(cond)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify_all()
        cond.release()
        self.assertReachesEventually(lambda: get_value(woken), 6)
        self.check_invariant(cond)

    def test_notify_n(self):
        cond = self.Condition()
        sleeping = self.Semaphore(0)
        woken = self.Semaphore(0)
        for i in range(3):
            p = self.Process(target=self.f, args=(cond, sleeping, woken))
            p.daemon = True
            p.start()
            self.addCleanup(p.join)
            t = threading.Thread(target=self.f, args=(cond, sleeping, woken))
            t.daemon = True
            t.start()
            self.addCleanup(t.join)
        for i in range(6):
            sleeping.acquire()
        time.sleep(DELTA)
        self.assertReturnsIfImplemented(0, get_value, woken)
        cond.acquire()
        cond.notify(n=2)
        cond.release()
        self.assertReachesEventually(lambda: get_value(woken), 2)
        cond.acquire()
        cond.notify(n=4)
        cond.release()
        self.assertReachesEventually(lambda: get_value(woken), 6)
        cond.acquire()
        cond.notify(n=3)
        cond.release()
        self.assertReturnsIfImplemented(6, get_value, woken)
        self.check_invariant(cond)

    def test_timeout(self):
        cond = self.Condition()
        wait = TimingWrapper(cond.wait)
        cond.acquire()
        res = wait(TIMEOUT1)
        cond.release()
        self.assertEqual(res, False)
        self.assertTimingAlmostEqual(wait.elapsed, TIMEOUT1)

    @classmethod
    def _test_waitfor_f(cls, cond, state):
        with cond:
            state.value = 0
            cond.notify()
            result = cond.wait_for(lambda: state.value == 4)
            if not result or state.value != 4:
                sys.exit(1)

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor(self):
        cond = self.Condition()
        state = self.Value('i', -1)
        p = self.Process(target=self._test_waitfor_f, args=(cond, state))
        p.daemon = True
        p.start()
        with cond:
            result = cond.wait_for(lambda: state.value == 0)
            self.assertTrue(result)
            self.assertEqual(state.value, 0)
        for i in range(4):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()
        join_process(p)
        self.assertEqual(p.exitcode, 0)

    @classmethod
    def _test_waitfor_timeout_f(cls, cond, state, success, sem):
        sem.release()
        with cond:
            expected = 0.1
            dt = time.monotonic()
            result = cond.wait_for(lambda: state.value == 4, timeout=expected)
            dt = time.monotonic() - dt
            if not result and expected * 0.6 < dt < expected * 10.0:
                success.value = True

    @unittest.skipUnless(HAS_SHAREDCTYPES, 'needs sharedctypes')
    def test_waitfor_timeout(self):
        cond = self.Condition()
        state = self.Value('i', 0)
        success = self.Value('i', False)
        sem = self.Semaphore(0)
        p = self.Process(target=self._test_waitfor_timeout_f, args=(cond, state, success, sem))
        p.daemon = True
        p.start()
        self.assertTrue(sem.acquire(timeout=support.LONG_TIMEOUT))
        for i in range(3):
            time.sleep(0.01)
            with cond:
                state.value += 1
                cond.notify()
        join_process(p)
        self.assertTrue(success.value)

    @classmethod
    def _test_wait_result(cls, c, pid):
        with c:
            c.notify()
        time.sleep(1)
        if pid is not None:
            os.kill(pid, signal.SIGINT)

    def test_wait_result(self):
        if isinstance(self, ProcessesMixin) and sys.platform != 'win32':
            pid = os.getpid()
        else:
            pid = None
        c = self.Condition()
        with c:
            self.assertFalse(c.wait(0))
            self.assertFalse(c.wait(0.1))
            p = self.Process(target=self._test_wait_result, args=(c, pid))
            p.start()
            self.assertTrue(c.wait(60))
            if pid is not None:
                self.assertRaises(KeyboardInterrupt, c.wait, 60)
            p.join()