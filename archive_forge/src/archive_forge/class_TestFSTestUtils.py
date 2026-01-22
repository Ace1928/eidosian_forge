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
class TestFSTestUtils(PsutilTestCase):

    def test_open_text(self):
        with open_text(__file__) as f:
            self.assertEqual(f.mode, 'r')

    def test_open_binary(self):
        with open_binary(__file__) as f:
            self.assertEqual(f.mode, 'rb')

    def test_safe_mkdir(self):
        testfn = self.get_testfn()
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)
        safe_mkdir(testfn)
        assert os.path.isdir(testfn)

    def test_safe_rmpath(self):
        testfn = self.get_testfn()
        open(testfn, 'w').close()
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        safe_rmpath(testfn)
        os.mkdir(testfn)
        safe_rmpath(testfn)
        assert not os.path.exists(testfn)
        with mock.patch('psutil.tests.os.stat', side_effect=OSError(errno.EINVAL, '')) as m:
            with self.assertRaises(OSError):
                safe_rmpath(testfn)
            assert m.called

    def test_chdir(self):
        testfn = self.get_testfn()
        base = os.getcwd()
        os.mkdir(testfn)
        with chdir(testfn):
            self.assertEqual(os.getcwd(), os.path.join(base, testfn))
        self.assertEqual(os.getcwd(), base)