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
@serialrun
class TestMemLeakClass(TestMemoryLeak):

    @retry_on_failure()
    def test_times(self):

        def fun():
            cnt['cnt'] += 1
        cnt = {'cnt': 0}
        self.execute(fun, times=10, warmup_times=15)
        self.assertEqual(cnt['cnt'], 26)

    def test_param_err(self):
        self.assertRaises(ValueError, self.execute, lambda: 0, times=0)
        self.assertRaises(ValueError, self.execute, lambda: 0, times=-1)
        self.assertRaises(ValueError, self.execute, lambda: 0, warmup_times=-1)
        self.assertRaises(ValueError, self.execute, lambda: 0, tolerance=-1)
        self.assertRaises(ValueError, self.execute, lambda: 0, retries=-1)

    @retry_on_failure()
    @unittest.skipIf(CI_TESTING, 'skipped on CI')
    @unittest.skipIf(COVERAGE, 'skipped during test coverage')
    def test_leak_mem(self):
        ls = []

        def fun(ls=ls):
            ls.append('x' * 124 * 1024)
        try:
            self.assertRaisesRegex(AssertionError, 'extra-mem', self.execute, fun, times=50)
        finally:
            del ls

    def test_unclosed_files(self):

        def fun():
            f = open(__file__)
            self.addCleanup(f.close)
            box.append(f)
        box = []
        kind = 'fd' if POSIX else 'handle'
        self.assertRaisesRegex(AssertionError, 'unclosed ' + kind, self.execute, fun)

    def test_tolerance(self):

        def fun():
            ls.append('x' * 24 * 1024)
        ls = []
        times = 100
        self.execute(fun, times=times, warmup_times=0, tolerance=200 * 1024 * 1024)
        self.assertEqual(len(ls), times + 1)

    def test_execute_w_exc(self):

        def fun_1():
            1 / 0
        self.execute_w_exc(ZeroDivisionError, fun_1)
        with self.assertRaises(ZeroDivisionError):
            self.execute_w_exc(OSError, fun_1)

        def fun_2():
            pass
        with self.assertRaises(AssertionError):
            self.execute_w_exc(ZeroDivisionError, fun_2)