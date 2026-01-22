import collections
import errno
import getpass
import itertools
import os
import signal
import socket
import stat
import subprocess
import sys
import textwrap
import time
import types
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import OSX
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._common import open_text
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil._compat import super
from psutil.tests import APPVEYOR
from psutil.tests import CI_TESTING
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_THREADS
from psutil.tests import MACOS_11PLUS
from psutil.tests import PYPY
from psutil.tests import PYTHON_EXE
from psutil.tests import PYTHON_EXE_ENV
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import copyload_shared_lib
from psutil.tests import create_c_exe
from psutil.tests import create_py_exe
from psutil.tests import mock
from psutil.tests import process_namespace
from psutil.tests import reap_children
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import skip_on_not_implemented
from psutil.tests import wait_for_pid
class TestPopen(PsutilTestCase):
    """Tests for psutil.Popen class."""

    @classmethod
    def tearDownClass(cls):
        reap_children()

    def test_misc(self):
        cmd = [PYTHON_EXE, '-c', 'import time; time.sleep(60);']
        with psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.name()
            proc.cpu_times()
            proc.stdin
            self.assertTrue(dir(proc))
            self.assertRaises(AttributeError, getattr, proc, 'foo')
            proc.terminate()
        if POSIX:
            self.assertEqual(proc.wait(5), -signal.SIGTERM)
        else:
            self.assertEqual(proc.wait(5), signal.SIGTERM)

    def test_ctx_manager(self):
        with psutil.Popen([PYTHON_EXE, '-V'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.communicate()
        assert proc.stdout.closed
        assert proc.stderr.closed
        assert proc.stdin.closed
        self.assertEqual(proc.returncode, 0)

    def test_kill_terminate(self):
        cmd = [PYTHON_EXE, '-c', 'import time; time.sleep(60);']
        with psutil.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=PYTHON_EXE_ENV) as proc:
            proc.terminate()
            proc.wait()
            self.assertRaises(psutil.NoSuchProcess, proc.terminate)
            self.assertRaises(psutil.NoSuchProcess, proc.kill)
            self.assertRaises(psutil.NoSuchProcess, proc.send_signal, signal.SIGTERM)
            if WINDOWS:
                self.assertRaises(psutil.NoSuchProcess, proc.send_signal, signal.CTRL_C_EVENT)
                self.assertRaises(psutil.NoSuchProcess, proc.send_signal, signal.CTRL_BREAK_EVENT)