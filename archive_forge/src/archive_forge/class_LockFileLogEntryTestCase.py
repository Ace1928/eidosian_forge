import doctest
import os
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock
from unittest.mock import patch
from zope.testing import setupstack
import zc.lockfile
class LockFileLogEntryTestCase(unittest.TestCase):
    """Tests for logging in case of lock failure"""

    def setUp(self):
        self.here = os.getcwd()
        self.tmp = tempfile.mkdtemp(prefix='zc.lockfile-test-')
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.here)
        setupstack.rmtree(self.tmp)

    def test_log_formatting(self):
        with patch('os.getpid', Mock(return_value=123)):
            with patch('socket.gethostname', Mock(return_value='myhostname')):
                lock = zc.lockfile.LockFile('f.lock', content_template='{pid}/{hostname}')
                with open('f.lock') as f:
                    self.assertEqual(' 123/myhostname\n', f.read())
                lock.close()

    def test_unlock_and_lock_while_multiprocessing_process_running(self):
        import multiprocessing
        lock = zc.lockfile.LockFile('l')
        q = multiprocessing.Queue()
        p = multiprocessing.Process(target=q.get)
        p.daemon = True
        p.start()
        lock.close()
        lock = zc.lockfile.LockFile('l')
        self.assertTrue(p.is_alive())
        q.put(0)
        lock.close()
        p.join()

    def test_simple_lock(self):
        assert isinstance(zc.lockfile.SimpleLockFile, type)
        lock = zc.lockfile.SimpleLockFile('s')
        with self.assertRaises(zc.lockfile.LockError):
            zc.lockfile.SimpleLockFile('s')
        lock.close()
        zc.lockfile.SimpleLockFile('s').close()