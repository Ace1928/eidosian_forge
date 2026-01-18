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
@unittest.skipIf(BSD, 'broken on BSD')
@unittest.skipIf(APPVEYOR, 'unreliable on APPVEYOR')
def test_open_files_2(self):
    p = psutil.Process()
    normcase = os.path.normcase
    testfn = self.get_testfn()
    with open(testfn, 'w') as fileobj:
        for file in p.open_files():
            if normcase(file.path) == normcase(fileobj.name) or file.fd == fileobj.fileno():
                break
        else:
            raise self.fail('no file found; files=%s' % repr(p.open_files()))
        self.assertEqual(normcase(file.path), normcase(fileobj.name))
        if WINDOWS:
            self.assertEqual(file.fd, -1)
        else:
            self.assertEqual(file.fd, fileobj.fileno())
        ntuple = p.open_files()[0]
        self.assertEqual(ntuple[0], ntuple.path)
        self.assertEqual(ntuple[1], ntuple.fd)
        self.assertNotIn(fileobj.name, p.open_files())