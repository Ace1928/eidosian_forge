import collections
import errno
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
from oslotest import base as test_base
from oslo_concurrency.fixture import lockutils as fixtures
from oslo_concurrency import lockutils
from oslo_config import fixture as config
class LockTestCase(test_base.BaseTestCase):

    def setUp(self):
        super(LockTestCase, self).setUp()
        self.config = self.useFixture(config.Config(lockutils.CONF)).config

    def test_synchronized_wrapped_function_metadata(self):

        @lockutils.synchronized('whatever', 'test-')
        def foo():
            """Bar."""
            pass
        self.assertEqual('Bar.', foo.__doc__, "Wrapped function's docstring got lost")
        self.assertEqual('foo', foo.__name__, "Wrapped function's name got mangled")

    def test_lock_internally_different_collections(self):
        s1 = lockutils.Semaphores()
        s2 = lockutils.Semaphores()
        trigger = threading.Event()
        who_ran = collections.deque()

        def f(name, semaphores, pull_trigger):
            with lockutils.internal_lock('testing', semaphores=semaphores):
                if pull_trigger:
                    trigger.set()
                else:
                    trigger.wait()
                who_ran.append(name)
        threads = [threading.Thread(target=f, args=(1, s1, True)), threading.Thread(target=f, args=(2, s2, False))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual([1, 2], sorted(who_ran))

    def test_lock_internally(self):
        """We can lock across multiple threads."""
        saved_sem_num = len(lockutils._semaphores)
        seen_threads = list()

        def f(_id):
            with lockutils.lock('testlock2', 'test-', external=False):
                for x in range(10):
                    seen_threads.append(_id)
        threads = []
        for i in range(10):
            thread = threading.Thread(target=f, args=(i,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(100, len(seen_threads))
        for i in range(10):
            for j in range(9):
                self.assertEqual(seen_threads[i * 10], seen_threads[i * 10 + 1 + j])
        self.assertEqual(saved_sem_num, len(lockutils._semaphores), 'Semaphore leak detected')

    def test_lock_internal_fair(self):
        """Check that we're actually fair."""

        def f(_id):
            with lockutils.lock('testlock', 'test-', external=False, fair=True):
                lock_holder.append(_id)
        lock_holder = []
        threads = []
        with lockutils.lock('testlock', 'test-', external=False, fair=True):
            for i in range(10):
                thread = threading.Thread(target=f, args=(i,))
                threads.append(thread)
                thread.start()
                time.sleep(0.5)
        for thread in threads:
            thread.join()
        self.assertEqual(10, len(lock_holder))
        for i in range(10):
            self.assertEqual(i, lock_holder[i])

    def test_fair_lock_with_semaphore(self):

        def do_test():
            s = lockutils.Semaphores()
            with lockutils.lock('testlock', 'test-', semaphores=s, fair=True):
                pass
        self.assertRaises(NotImplementedError, do_test)

    def test_fair_lock_with_nonblocking(self):

        def do_test():
            with lockutils.lock('testlock', 'test-', fair=True, blocking=False):
                pass
        self.assertRaises(NotImplementedError, do_test)

    def test_nested_synchronized_external_works(self):
        """We can nest external syncs."""
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
        sentinel = object()

        @lockutils.synchronized('testlock1', 'test-', external=True)
        def outer_lock():

            @lockutils.synchronized('testlock2', 'test-', external=True)
            def inner_lock():
                return sentinel
            return inner_lock()
        self.assertEqual(sentinel, outer_lock())

    def _do_test_lock_externally(self):
        """We can lock across multiple processes."""
        children = []
        for n in range(50):
            queue = multiprocessing.Queue()
            proc = multiprocessing.Process(target=lock_files, args=(tempfile.mkdtemp(), queue))
            proc.start()
            children.append((proc, queue))
        for child, queue in children:
            child.join()
            count = queue.get(block=False)
            self.assertEqual(50, count)

    def test_lock_externally(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
        self._do_test_lock_externally()

    def test_lock_externally_lock_dir_not_exist(self):
        lock_dir = tempfile.mkdtemp()
        os.rmdir(lock_dir)
        self.config(lock_path=lock_dir, group='oslo_concurrency')
        self._do_test_lock_externally()

    def test_lock_with_prefix(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
        foo = lockutils.lock_with_prefix('mypfix-')
        with foo('mylock', external=True):
            pass

    def test_synchronized_with_prefix(self):
        lock_name = 'mylock'
        lock_pfix = 'mypfix-'
        foo = lockutils.synchronized_with_prefix(lock_pfix)

        @foo(lock_name, external=True)
        def bar(dirpath, pfix, name):
            return True
        lock_dir = tempfile.mkdtemp()
        self.config(lock_path=lock_dir, group='oslo_concurrency')
        self.assertTrue(bar(lock_dir, lock_pfix, lock_name))

    def test_synchronized_without_prefix(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')

        @lockutils.synchronized('lock', external=True)
        def test_without_prefix():
            pass
        test_without_prefix()

    def test_synchronized_prefix_without_hypen(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')

        @lockutils.synchronized('lock', 'hypen', True)
        def test_without_hypen():
            pass
        test_without_hypen()

    def test_contextlock(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
        with lockutils.lock('test') as sem:
            self.assertIsInstance(sem, threading.Semaphore)
            with lockutils.lock('test2', external=True) as lock:
                self.assertTrue(lock.exists())
            with lockutils.lock('test1', external=True) as lock1:
                self.assertIsInstance(lock1, lockutils.InterProcessLock)

    def test_contextlock_unlocks(self):
        self.config(lock_path=tempfile.mkdtemp(), group='oslo_concurrency')
        with lockutils.lock('test') as sem:
            self.assertIsInstance(sem, threading.Semaphore)
            with lockutils.lock('test2', external=True) as lock:
                self.assertTrue(lock.exists())
            with lockutils.lock('test2', external=True) as lock:
                self.assertTrue(lock.exists())
        with lockutils.lock('test') as sem2:
            self.assertEqual(sem, sem2)

    @mock.patch('logging.Logger.info')
    @mock.patch('os.remove')
    @mock.patch('oslo_concurrency.lockutils._get_lock_path')
    def test_remove_lock_external_file_exists(self, path_mock, remove_mock, log_mock):
        lockutils.remove_external_lock_file(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        path_mock.assert_called_once_with(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        remove_mock.assert_called_once_with(path_mock.return_value)
        log_mock.assert_not_called()

    @mock.patch('logging.Logger.warning')
    @mock.patch('os.remove', side_effect=OSError(errno.ENOENT, None))
    @mock.patch('oslo_concurrency.lockutils._get_lock_path')
    def test_remove_lock_external_file_doesnt_exists(self, path_mock, remove_mock, log_mock):
        lockutils.remove_external_lock_file(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        path_mock.assert_called_once_with(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        remove_mock.assert_called_once_with(path_mock.return_value)
        log_mock.assert_not_called()

    @mock.patch('logging.Logger.warning')
    @mock.patch('os.remove', side_effect=OSError(errno.EPERM, None))
    @mock.patch('oslo_concurrency.lockutils._get_lock_path')
    def test_remove_lock_external_file_permission_error(self, path_mock, remove_mock, log_mock):
        lockutils.remove_external_lock_file(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        path_mock.assert_called_once_with(mock.sentinel.name, mock.sentinel.prefix, mock.sentinel.lock_path)
        remove_mock.assert_called_once_with(path_mock.return_value)
        log_mock.assert_called()

    def test_no_slash_in_b64(self):
        with lockutils.lock('foobar'):
            pass