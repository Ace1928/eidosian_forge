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
def test_reused_pid(self):
    subp = self.spawn_testproc()
    p = psutil.Process(subp.pid)
    p._ident = (p.pid, p.create_time() + 100)
    assert not p.is_running()
    assert p != psutil.Process(subp.pid)
    msg = 'process no longer exists and its PID has been reused'
    ns = process_namespace(p)
    for fun, name in ns.iter(ns.setters + ns.killers, clear_cache=False):
        with self.subTest(name=name):
            self.assertRaisesRegex(psutil.NoSuchProcess, msg, fun)
    self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.ppid)
    self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.parent)
    self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.parents)
    self.assertRaisesRegex(psutil.NoSuchProcess, msg, p.children)