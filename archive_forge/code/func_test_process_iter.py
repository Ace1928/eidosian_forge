import contextlib
import datetime
import errno
import os
import platform
import pprint
import shutil
import signal
import socket
import sys
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import long
from psutil.tests import ASCII_FS
from psutil.tests import CI_TESTING
from psutil.tests import DEVNULL
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import IS_64BIT
from psutil.tests import MACOS_12PLUS
from psutil.tests import PYPY
from psutil.tests import UNICODE_SUFFIX
from psutil.tests import PsutilTestCase
from psutil.tests import check_net_address
from psutil.tests import enum
from psutil.tests import mock
from psutil.tests import retry_on_failure
def test_process_iter(self):
    self.assertIn(os.getpid(), [x.pid for x in psutil.process_iter()])
    sproc = self.spawn_testproc()
    self.assertIn(sproc.pid, [x.pid for x in psutil.process_iter()])
    p = psutil.Process(sproc.pid)
    p.kill()
    p.wait()
    self.assertNotIn(sproc.pid, [x.pid for x in psutil.process_iter()])
    ls = [x for x in psutil.process_iter()]
    self.assertEqual(sorted(ls, key=lambda x: x.pid), sorted(set(ls), key=lambda x: x.pid))
    with mock.patch('psutil.Process', side_effect=psutil.NoSuchProcess(os.getpid())):
        self.assertEqual(list(psutil.process_iter()), [])
    with mock.patch('psutil.Process', side_effect=psutil.AccessDenied(os.getpid())):
        with self.assertRaises(psutil.AccessDenied):
            list(psutil.process_iter())