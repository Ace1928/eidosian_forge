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
class _TestQueue(BaseTestCase):

    @classmethod
    def _test_put(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        for i in range(6):
            queue.get()
        parent_can_continue.set()

    def test_put(self):
        MAXSIZE = 6
        queue = self.Queue(maxsize=MAXSIZE)
        child_can_start = self.Event()
        parent_can_continue = self.Event()
        proc = self.Process(target=self._test_put, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)
        queue.put(1)
        queue.put(2, True)
        queue.put(3, True, None)
        queue.put(4, False)
        queue.put(5, False, None)
        queue.put_nowait(6)
        time.sleep(DELTA)
        self.assertEqual(queue_empty(queue), False)
        self.assertEqual(queue_full(queue, MAXSIZE), True)
        put = TimingWrapper(queue.put)
        put_nowait = TimingWrapper(queue.put_nowait)
        self.assertRaises(pyqueue.Full, put, 7, False)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, False, None)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put_nowait, 7)
        self.assertTimingAlmostEqual(put_nowait.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, True, TIMEOUT1)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT1)
        self.assertRaises(pyqueue.Full, put, 7, False, TIMEOUT2)
        self.assertTimingAlmostEqual(put.elapsed, 0)
        self.assertRaises(pyqueue.Full, put, 7, True, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(put.elapsed, TIMEOUT3)
        child_can_start.set()
        parent_can_continue.wait()
        self.assertEqual(queue_empty(queue), True)
        self.assertEqual(queue_full(queue, MAXSIZE), False)
        proc.join()
        close_queue(queue)

    @classmethod
    def _test_get(cls, queue, child_can_start, parent_can_continue):
        child_can_start.wait()
        queue.put(2)
        queue.put(3)
        queue.put(4)
        queue.put(5)
        parent_can_continue.set()

    def test_get(self):
        queue = self.Queue()
        child_can_start = self.Event()
        parent_can_continue = self.Event()
        proc = self.Process(target=self._test_get, args=(queue, child_can_start, parent_can_continue))
        proc.daemon = True
        proc.start()
        self.assertEqual(queue_empty(queue), True)
        child_can_start.set()
        parent_can_continue.wait()
        time.sleep(DELTA)
        self.assertEqual(queue_empty(queue), False)
        self.assertEqual(queue.get(True, None), 2)
        self.assertEqual(queue.get(True), 3)
        self.assertEqual(queue.get(timeout=1), 4)
        self.assertEqual(queue.get_nowait(), 5)
        self.assertEqual(queue_empty(queue), True)
        get = TimingWrapper(queue.get)
        get_nowait = TimingWrapper(queue.get_nowait)
        self.assertRaises(pyqueue.Empty, get, False)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, False, None)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get_nowait)
        self.assertTimingAlmostEqual(get_nowait.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, True, TIMEOUT1)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)
        self.assertRaises(pyqueue.Empty, get, False, TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, 0)
        self.assertRaises(pyqueue.Empty, get, timeout=TIMEOUT3)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT3)
        proc.join()
        close_queue(queue)

    @classmethod
    def _test_fork(cls, queue):
        for i in range(10, 20):
            queue.put(i)

    def test_fork(self):
        queue = self.Queue()
        for i in range(10):
            queue.put(i)
        time.sleep(DELTA)
        p = self.Process(target=self._test_fork, args=(queue,))
        p.daemon = True
        p.start()
        for i in range(20):
            self.assertEqual(queue.get(), i)
        self.assertRaises(pyqueue.Empty, queue.get, False)
        p.join()
        close_queue(queue)

    def test_qsize(self):
        q = self.Queue()
        try:
            self.assertEqual(q.qsize(), 0)
        except NotImplementedError:
            self.skipTest('qsize method not implemented')
        q.put(1)
        self.assertEqual(q.qsize(), 1)
        q.put(5)
        self.assertEqual(q.qsize(), 2)
        q.get()
        self.assertEqual(q.qsize(), 1)
        q.get()
        self.assertEqual(q.qsize(), 0)
        close_queue(q)

    @classmethod
    def _test_task_done(cls, q):
        for obj in iter(q.get, None):
            time.sleep(DELTA)
            q.task_done()

    def test_task_done(self):
        queue = self.JoinableQueue()
        workers = [self.Process(target=self._test_task_done, args=(queue,)) for i in range(4)]
        for p in workers:
            p.daemon = True
            p.start()
        for i in range(10):
            queue.put(i)
        queue.join()
        for p in workers:
            queue.put(None)
        for p in workers:
            p.join()
        close_queue(queue)

    def test_no_import_lock_contention(self):
        with os_helper.temp_cwd():
            module_name = 'imported_by_an_imported_module'
            with open(module_name + '.py', 'w', encoding='utf-8') as f:
                f.write("if 1:\n                    import multiprocess as multiprocessing\n\n                    q = multiprocessing.Queue()\n                    q.put('knock knock')\n                    q.get(timeout=3)\n                    q.close()\n                    del q\n                ")
            with import_helper.DirsOnSysPath(os.getcwd()):
                try:
                    __import__(module_name)
                except pyqueue.Empty:
                    self.fail('Probable regression on import lock contention; see Issue #22853')

    def test_timeout(self):
        q = multiprocessing.Queue()
        start = time.monotonic()
        self.assertRaises(pyqueue.Empty, q.get, True, 0.2)
        delta = time.monotonic() - start
        self.assertGreaterEqual(delta, 0.1)
        close_queue(q)

    def test_queue_feeder_donot_stop_onexc(self):
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):

            def __reduce__(self):
                raise AttributeError
        with test.support.captured_stderr():
            q = self.Queue()
            q.put(NotSerializable())
            q.put(True)
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
            close_queue(q)
        with test.support.captured_stderr():
            q = self.Queue(maxsize=1)
            q.put(NotSerializable())
            q.put(True)
            try:
                self.assertEqual(q.qsize(), 1)
            except NotImplementedError:
                pass
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
            self.assertTrue(q.empty())
            close_queue(q)

    def test_queue_feeder_on_queue_feeder_error(self):
        if self.TYPE != 'processes':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class NotSerializable(object):
            """Mock unserializable object"""

            def __init__(self):
                self.reduce_was_called = False
                self.on_queue_feeder_error_was_called = False

            def __reduce__(self):
                self.reduce_was_called = True
                raise AttributeError

        class SafeQueue(multiprocessing.queues.Queue):
            """Queue with overloaded _on_queue_feeder_error hook"""

            @staticmethod
            def _on_queue_feeder_error(e, obj):
                if isinstance(e, AttributeError) and isinstance(obj, NotSerializable):
                    obj.on_queue_feeder_error_was_called = True
        not_serializable_obj = NotSerializable()
        with test.support.captured_stderr():
            q = SafeQueue(ctx=multiprocessing.get_context())
            q.put(not_serializable_obj)
            q.put(True)
            self.assertTrue(q.get(timeout=support.SHORT_TIMEOUT))
        self.assertTrue(not_serializable_obj.reduce_was_called)
        self.assertTrue(not_serializable_obj.on_queue_feeder_error_was_called)

    def test_closed_queue_put_get_exceptions(self):
        for q in (multiprocessing.Queue(), multiprocessing.JoinableQueue()):
            q.close()
            with self.assertRaisesRegex(ValueError, 'is closed'):
                q.put('foo')
            with self.assertRaisesRegex(ValueError, 'is closed'):
                q.get()