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
@hashlib_helper.requires_hashdigest('md5')
class TestSyncManagerTypes(unittest.TestCase):
    """Test all the types which can be shared between a parent and a
    child process by using a manager which acts as an intermediary
    between them.

    In the following unit-tests the base type is created in the parent
    process, the @classmethod represents the worker process and the
    shared object is readable and editable between the two.

    # The child.
    @classmethod
    def _test_list(cls, obj):
        assert obj[0] == 5
        assert obj.append(6)

    # The parent.
    def test_list(self):
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert o[1] == 6
    """
    manager_class = multiprocessing.managers.SyncManager

    def setUp(self):
        self.manager = self.manager_class()
        self.manager.start()
        self.proc = None

    def tearDown(self):
        if self.proc is not None and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join()
        self.manager.shutdown()
        self.manager = None
        self.proc = None

    @classmethod
    def setUpClass(cls):
        support.reap_children()
    tearDownClass = setUpClass

    def wait_proc_exit(self):
        join_process(self.proc)
        start_time = time.monotonic()
        t = 0.01
        while len(multiprocessing.active_children()) > 1:
            time.sleep(t)
            t *= 2
            dt = time.monotonic() - start_time
            if dt >= 5.0:
                test.support.environment_altered = True
                support.print_warning(f'multiprocess.Manager still has {multiprocessing.active_children()} active children after {dt} seconds')
                break

    def run_worker(self, worker, obj):
        self.proc = multiprocessing.Process(target=worker, args=(obj,))
        self.proc.daemon = True
        self.proc.start()
        self.wait_proc_exit()
        self.assertEqual(self.proc.exitcode, 0)

    @classmethod
    def _test_event(cls, obj):
        assert obj.is_set()
        obj.wait()
        obj.clear()
        obj.wait(0.001)

    def test_event(self):
        o = self.manager.Event()
        o.set()
        self.run_worker(self._test_event, o)
        assert not o.is_set()
        o.wait(0.001)

    @classmethod
    def _test_lock(cls, obj):
        obj.acquire()

    def test_lock(self, lname='Lock'):
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_lock, o)
        o.release()
        self.assertRaises(RuntimeError, o.release)

    @classmethod
    def _test_rlock(cls, obj):
        obj.acquire()
        obj.release()

    def test_rlock(self, lname='Lock'):
        o = getattr(self.manager, lname)()
        self.run_worker(self._test_rlock, o)

    @classmethod
    def _test_semaphore(cls, obj):
        obj.acquire()

    def test_semaphore(self, sname='Semaphore'):
        o = getattr(self.manager, sname)()
        self.run_worker(self._test_semaphore, o)
        o.release()

    def test_bounded_semaphore(self):
        self.test_semaphore(sname='BoundedSemaphore')

    @classmethod
    def _test_condition(cls, obj):
        obj.acquire()
        obj.release()

    def test_condition(self):
        o = self.manager.Condition()
        self.run_worker(self._test_condition, o)

    @classmethod
    def _test_barrier(cls, obj):
        assert obj.parties == 5
        obj.reset()

    def test_barrier(self):
        o = self.manager.Barrier(5)
        self.run_worker(self._test_barrier, o)

    @classmethod
    def _test_pool(cls, obj):
        with obj:
            pass

    def test_pool(self):
        o = self.manager.Pool(processes=4)
        self.run_worker(self._test_pool, o)

    @classmethod
    def _test_queue(cls, obj):
        assert obj.qsize() == 2
        assert obj.full()
        assert not obj.empty()
        assert obj.get() == 5
        assert not obj.empty()
        assert obj.get() == 6
        assert obj.empty()

    def test_queue(self, qname='Queue'):
        o = getattr(self.manager, qname)(2)
        o.put(5)
        o.put(6)
        self.run_worker(self._test_queue, o)
        assert o.empty()
        assert not o.full()

    def test_joinable_queue(self):
        self.test_queue('JoinableQueue')

    @classmethod
    def _test_list(cls, obj):
        assert obj[0] == 5
        assert obj.count(5) == 1
        assert obj.index(5) == 0
        obj.sort()
        obj.reverse()
        for x in obj:
            pass
        assert len(obj) == 1
        assert obj.pop(0) == 5

    def test_list(self):
        o = self.manager.list()
        o.append(5)
        self.run_worker(self._test_list, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_dict(cls, obj):
        assert len(obj) == 1
        assert obj['foo'] == 5
        assert obj.get('foo') == 5
        assert list(obj.items()) == [('foo', 5)]
        assert list(obj.keys()) == ['foo']
        assert list(obj.values()) == [5]
        assert obj.copy() == {'foo': 5}
        assert obj.popitem() == ('foo', 5)

    def test_dict(self):
        o = self.manager.dict()
        o['foo'] = 5
        self.run_worker(self._test_dict, o)
        assert not o
        self.assertEqual(len(o), 0)

    @classmethod
    def _test_value(cls, obj):
        assert obj.value == 1
        assert obj.get() == 1
        obj.set(2)

    def test_value(self):
        o = self.manager.Value('i', 1)
        self.run_worker(self._test_value, o)
        self.assertEqual(o.value, 2)
        self.assertEqual(o.get(), 2)

    @classmethod
    def _test_array(cls, obj):
        assert obj[0] == 0
        assert obj[1] == 1
        assert len(obj) == 2
        assert list(obj) == [0, 1]

    def test_array(self):
        o = self.manager.Array('i', [0, 1])
        self.run_worker(self._test_array, o)

    @classmethod
    def _test_namespace(cls, obj):
        assert obj.x == 0
        assert obj.y == 1

    def test_namespace(self):
        o = self.manager.Namespace()
        o.x = 0
        o.y = 1
        self.run_worker(self._test_namespace, o)