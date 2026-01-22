import collections
import contextlib
import errno
import os
import socket
import stat
import subprocess
import unittest
import psutil
import psutil.tests
from psutil import FREEBSD
from psutil import NETBSD
from psutil import POSIX
from psutil._common import open_binary
from psutil._common import open_text
from psutil._common import supports_ipv6
from psutil.tests import CI_TESTING
from psutil.tests import COVERAGE
from psutil.tests import HAS_CONNECTIONS_UNIX
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import TestMemoryLeak
from psutil.tests import bind_socket
from psutil.tests import bind_unix_socket
from psutil.tests import call_until
from psutil.tests import chdir
from psutil.tests import create_sockets
from psutil.tests import filter_proc_connections
from psutil.tests import get_free_port
from psutil.tests import is_namedtuple
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry
from psutil.tests import retry_on_failure
from psutil.tests import safe_mkdir
from psutil.tests import safe_rmpath
from psutil.tests import serialrun
from psutil.tests import system_namespace
from psutil.tests import tcp_socketpair
from psutil.tests import terminate
from psutil.tests import unix_socketpair
from psutil.tests import wait_for_file
from psutil.tests import wait_for_pid
class TestSyncTestUtils(PsutilTestCase):

    def test_wait_for_pid(self):
        wait_for_pid(os.getpid())
        nopid = max(psutil.pids()) + 99999
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            self.assertRaises(psutil.NoSuchProcess, wait_for_pid, nopid)

    def test_wait_for_file(self):
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn)
        assert not os.path.exists(testfn)

    def test_wait_for_file_empty(self):
        testfn = self.get_testfn()
        with open(testfn, 'w'):
            pass
        wait_for_file(testfn, empty=True)
        assert not os.path.exists(testfn)

    def test_wait_for_file_no_file(self):
        testfn = self.get_testfn()
        with mock.patch('psutil.tests.retry.__iter__', return_value=iter([0])):
            self.assertRaises(IOError, wait_for_file, testfn)

    def test_wait_for_file_no_delete(self):
        testfn = self.get_testfn()
        with open(testfn, 'w') as f:
            f.write('foo')
        wait_for_file(testfn, delete=False)
        assert os.path.exists(testfn)

    def test_call_until(self):
        ret = call_until(lambda: 1, 'ret == 1')
        self.assertEqual(ret, 1)