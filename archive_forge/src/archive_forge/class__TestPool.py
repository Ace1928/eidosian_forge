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
class _TestPool(BaseTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.pool = cls.Pool(4)

    @classmethod
    def tearDownClass(cls):
        cls.pool.terminate()
        cls.pool.join()
        cls.pool = None
        super().tearDownClass()

    def test_apply(self):
        papply = self.pool.apply
        self.assertEqual(papply(sqr, (5,)), sqr(5))
        self.assertEqual(papply(sqr, (), {'x': 3}), sqr(x=3))

    def test_map(self):
        pmap = self.pool.map
        self.assertEqual(pmap(sqr, list(range(10))), list(map(sqr, list(range(10)))))
        self.assertEqual(pmap(sqr, list(range(100)), chunksize=20), list(map(sqr, list(range(100)))))

    def test_starmap(self):
        psmap = self.pool.starmap
        tuples = list(zip(range(10), range(9, -1, -1)))
        self.assertEqual(psmap(mul, tuples), list(itertools.starmap(mul, tuples)))
        tuples = list(zip(range(100), range(99, -1, -1)))
        self.assertEqual(psmap(mul, tuples, chunksize=20), list(itertools.starmap(mul, tuples)))

    def test_starmap_async(self):
        tuples = list(zip(range(100), range(99, -1, -1)))
        self.assertEqual(self.pool.starmap_async(mul, tuples).get(), list(itertools.starmap(mul, tuples)))

    def test_map_async(self):
        self.assertEqual(self.pool.map_async(sqr, list(range(10))).get(), list(map(sqr, list(range(10)))))

    def test_map_async_callbacks(self):
        call_args = self.manager.list() if self.TYPE == 'manager' else []
        self.pool.map_async(int, ['1'], callback=call_args.append, error_callback=call_args.append).wait()
        self.assertEqual(1, len(call_args))
        self.assertEqual([1], call_args[0])
        self.pool.map_async(int, ['a'], callback=call_args.append, error_callback=call_args.append).wait()
        self.assertEqual(2, len(call_args))
        self.assertIsInstance(call_args[1], ValueError)

    def test_map_unplicklable(self):
        if self.TYPE == 'threads':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))

        class A(object):

            def __reduce__(self):
                raise RuntimeError('cannot pickle')
        with self.assertRaises(RuntimeError):
            self.pool.map(sqr, [A()] * 10)

    def test_map_chunksize(self):
        try:
            self.pool.map_async(sqr, [], chunksize=1).get(timeout=TIMEOUT1)
        except multiprocessing.TimeoutError:
            self.fail('pool.map_async with chunksize stalled on null list')

    def test_map_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(1, -1), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, exception_throwing_generator(10, 3), 1)

        class SpecialIterable:

            def __iter__(self):
                return self

            def __next__(self):
                raise SayWhenError

            def __len__(self):
                return 1
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)
        with self.assertRaises(SayWhenError):
            self.pool.map(sqr, SpecialIterable(), 1)

    def test_async(self):
        res = self.pool.apply_async(sqr, (7, TIMEOUT1))
        get = TimingWrapper(res.get)
        self.assertEqual(get(), 49)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT1)

    def test_async_timeout(self):
        res = self.pool.apply_async(sqr, (6, TIMEOUT2 + 1.0))
        get = TimingWrapper(res.get)
        self.assertRaises(multiprocessing.TimeoutError, get, timeout=TIMEOUT2)
        self.assertTimingAlmostEqual(get.elapsed, TIMEOUT2)

    def test_imap(self):
        it = self.pool.imap(sqr, list(range(10)))
        self.assertEqual(list(it), list(map(sqr, list(range(10)))))
        it = self.pool.imap(sqr, list(range(10)))
        for i in range(10):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, it.__next__)
        it = self.pool.imap(sqr, list(range(1000)), chunksize=100)
        for i in range(1000):
            self.assertEqual(next(it), i * i)
        self.assertRaises(StopIteration, it.__next__)

    def test_imap_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(10, 3), 1)
        for i in range(3):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 2)
        for i in range(6):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap(sqr, exception_throwing_generator(20, 7), 4)
        for i in range(4):
            self.assertEqual(next(it), i * i)
        self.assertRaises(SayWhenError, it.__next__)

    def test_imap_unordered(self):
        it = self.pool.imap_unordered(sqr, list(range(10)))
        self.assertEqual(sorted(it), list(map(sqr, list(range(10)))))
        it = self.pool.imap_unordered(sqr, list(range(1000)), chunksize=100)
        self.assertEqual(sorted(it), list(map(sqr, list(range(1000)))))

    def test_imap_unordered_handle_iterable_exception(self):
        if self.TYPE == 'manager':
            self.skipTest('test not appropriate for {}'.format(self.TYPE))
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(1, -1), 1)
        self.assertRaises(SayWhenError, it.__next__)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(10, 3), 1)
        expected_values = list(map(sqr, list(range(10))))
        with self.assertRaises(SayWhenError):
            for i in range(10):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)
        it = self.pool.imap_unordered(sqr, exception_throwing_generator(20, 7), 2)
        expected_values = list(map(sqr, list(range(20))))
        with self.assertRaises(SayWhenError):
            for i in range(20):
                value = next(it)
                self.assertIn(value, expected_values)
                expected_values.remove(value)

    def test_make_pool(self):
        expected_error = RemoteError if self.TYPE == 'manager' else ValueError
        self.assertRaises(expected_error, self.Pool, -1)
        self.assertRaises(expected_error, self.Pool, 0)
        if self.TYPE != 'manager':
            p = self.Pool(3)
            try:
                self.assertEqual(3, len(p._pool))
            finally:
                p.close()
                p.join()

    def test_terminate(self):
        result = self.pool.map_async(time.sleep, [0.1 for i in range(10000)], chunksize=1)
        self.pool.terminate()
        join = TimingWrapper(self.pool.join)
        join()
        self.assertLess(join.elapsed, 2.0)

    def test_empty_iterable(self):
        p = self.Pool(1)
        self.assertEqual(p.map(sqr, []), [])
        self.assertEqual(list(p.imap(sqr, [])), [])
        self.assertEqual(list(p.imap_unordered(sqr, [])), [])
        self.assertEqual(p.map_async(sqr, []).get(), [])
        p.close()
        p.join()

    def test_context(self):
        if self.TYPE == 'processes':
            L = list(range(10))
            expected = [sqr(i) for i in L]
            with self.Pool(2) as p:
                r = p.map_async(sqr, L)
                self.assertEqual(r.get(), expected)
            p.join()
            self.assertRaises(ValueError, p.map_async, sqr, L)

    @classmethod
    def _test_traceback(cls):
        raise RuntimeError(123)

    @unittest.skipIf(True, 'fails with is_dill(obj, child=True)')
    def test_traceback(self):
        if self.TYPE == 'processes':
            with self.Pool(1) as p:
                try:
                    p.apply(self._test_traceback)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected RuntimeError')
            p.join()
            self.assertIs(type(exc), RuntimeError)
            self.assertEqual(exc.args, (123,))
            cause = exc.__cause__
            self.assertIs(type(cause), multiprocessing.pool.RemoteTraceback)
            self.assertIn('raise RuntimeError(123) # some comment', cause.tb)
            with test.support.captured_stderr() as f1:
                try:
                    raise exc
                except RuntimeError:
                    sys.excepthook(*sys.exc_info())
            self.assertIn('raise RuntimeError(123) # some comment', f1.getvalue())
            with self.Pool(1) as p:
                try:
                    p.map(sqr, exception_throwing_generator(1, -1), 1)
                except Exception as e:
                    exc = e
                else:
                    self.fail('expected SayWhenError')
                self.assertIs(type(exc), SayWhenError)
                self.assertIs(exc.__cause__, None)
            p.join()

    @classmethod
    def _test_wrapped_exception(cls):
        raise RuntimeError('foo')

    @unittest.skipIf(True, 'fails with is_dill(obj, child=True)')
    def test_wrapped_exception(self):
        with self.Pool(1) as p:
            with self.assertRaises(RuntimeError):
                p.apply(self._test_wrapped_exception)
        p.join()

    def test_map_no_failfast(self):
        t_start = time.monotonic()
        with self.assertRaises(ValueError):
            with self.Pool(2) as p:
                try:
                    p.map(raise_large_valuerror, [0, 1])
                finally:
                    time.sleep(0.5)
                    p.close()
                    p.join()
        self.assertGreater(time.monotonic() - t_start, 0.9)

    def test_release_task_refs(self):
        objs = [CountedObject() for i in range(10)]
        refs = [weakref.ref(o) for o in objs]
        self.pool.map(identity, objs)
        del objs
        gc.collect()
        time.sleep(DELTA)
        self.assertEqual(set((wr() for wr in refs)), {None})
        self.assertEqual(CountedObject.n_instances, 0)

    def test_enter(self):
        if self.TYPE == 'manager':
            self.skipTest('test not applicable to manager')
        pool = self.Pool(1)
        with pool:
            pass
        with self.assertRaises(ValueError):
            with pool:
                pass
        pool.join()

    def test_resource_warning(self):
        if self.TYPE == 'manager':
            self.skipTest('test not applicable to manager')
        pool = self.Pool(1)
        pool.terminate()
        pool.join()
        pool._state = multiprocessing.pool.RUN
        with warnings_helper.check_warnings(('unclosed running multiprocessing pool', ResourceWarning)):
            pool = None
            support.gc_collect()