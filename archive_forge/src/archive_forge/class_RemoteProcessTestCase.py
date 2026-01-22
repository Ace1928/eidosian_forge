import datetime
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import warnings
import psutil
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import super
from psutil._compat import which
from psutil.tests import APPVEYOR
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_BATTERY
from psutil.tests import IS_64BIT
from psutil.tests import PY3
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
@unittest.skipIf(not WINDOWS, 'WINDOWS only')
class RemoteProcessTestCase(PsutilTestCase):
    """Certain functions require calling ReadProcessMemory.
    This trivially works when called on the current process.
    Check that this works on other processes, especially when they
    have a different bitness.
    """

    @staticmethod
    def find_other_interpreter():
        code = 'import sys; sys.stdout.write(str(sys.maxsize > 2**32))'
        for filename in glob.glob('C:\\Python*\\python.exe'):
            proc = subprocess.Popen(args=[filename, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output, _ = proc.communicate()
            proc.wait()
            if output == str(not IS_64BIT):
                return filename
    test_args = ['-c', 'import sys; sys.stdin.read()']

    def setUp(self):
        super().setUp()
        other_python = self.find_other_interpreter()
        if other_python is None:
            raise unittest.SkipTest('could not find interpreter with opposite bitness')
        if IS_64BIT:
            self.python64 = sys.executable
            self.python32 = other_python
        else:
            self.python64 = other_python
            self.python32 = sys.executable
        env = os.environ.copy()
        env['THINK_OF_A_NUMBER'] = str(os.getpid())
        self.proc32 = self.spawn_testproc([self.python32] + self.test_args, env=env, stdin=subprocess.PIPE)
        self.proc64 = self.spawn_testproc([self.python64] + self.test_args, env=env, stdin=subprocess.PIPE)

    def tearDown(self):
        super().tearDown()
        self.proc32.communicate()
        self.proc64.communicate()

    def test_cmdline_32(self):
        p = psutil.Process(self.proc32.pid)
        self.assertEqual(len(p.cmdline()), 3)
        self.assertEqual(p.cmdline()[1:], self.test_args)

    def test_cmdline_64(self):
        p = psutil.Process(self.proc64.pid)
        self.assertEqual(len(p.cmdline()), 3)
        self.assertEqual(p.cmdline()[1:], self.test_args)

    def test_cwd_32(self):
        p = psutil.Process(self.proc32.pid)
        self.assertEqual(p.cwd(), os.getcwd())

    def test_cwd_64(self):
        p = psutil.Process(self.proc64.pid)
        self.assertEqual(p.cwd(), os.getcwd())

    def test_environ_32(self):
        p = psutil.Process(self.proc32.pid)
        e = p.environ()
        self.assertIn('THINK_OF_A_NUMBER', e)
        self.assertEqual(e['THINK_OF_A_NUMBER'], str(os.getpid()))

    def test_environ_64(self):
        p = psutil.Process(self.proc64.pid)
        try:
            p.environ()
        except psutil.AccessDenied:
            pass