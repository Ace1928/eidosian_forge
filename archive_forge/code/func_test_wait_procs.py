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
@unittest.skipIf(PYPY and WINDOWS, 'spawn_testproc() unreliable on PYPY + WINDOWS')
def test_wait_procs(self):

    def callback(p):
        pids.append(p.pid)
    pids = []
    sproc1 = self.spawn_testproc()
    sproc2 = self.spawn_testproc()
    sproc3 = self.spawn_testproc()
    procs = [psutil.Process(x.pid) for x in (sproc1, sproc2, sproc3)]
    self.assertRaises(ValueError, psutil.wait_procs, procs, timeout=-1)
    self.assertRaises(TypeError, psutil.wait_procs, procs, callback=1)
    t = time.time()
    gone, alive = psutil.wait_procs(procs, timeout=0.01, callback=callback)
    self.assertLess(time.time() - t, 0.5)
    self.assertEqual(gone, [])
    self.assertEqual(len(alive), 3)
    self.assertEqual(pids, [])
    for p in alive:
        self.assertFalse(hasattr(p, 'returncode'))

    @retry_on_failure(30)
    def test_1(procs, callback):
        gone, alive = psutil.wait_procs(procs, timeout=0.03, callback=callback)
        self.assertEqual(len(gone), 1)
        self.assertEqual(len(alive), 2)
        return (gone, alive)
    sproc3.terminate()
    gone, alive = test_1(procs, callback)
    self.assertIn(sproc3.pid, [x.pid for x in gone])
    if POSIX:
        self.assertEqual(gone.pop().returncode, -signal.SIGTERM)
    else:
        self.assertEqual(gone.pop().returncode, 1)
    self.assertEqual(pids, [sproc3.pid])
    for p in alive:
        self.assertFalse(hasattr(p, 'returncode'))

    @retry_on_failure(30)
    def test_2(procs, callback):
        gone, alive = psutil.wait_procs(procs, timeout=0.03, callback=callback)
        self.assertEqual(len(gone), 3)
        self.assertEqual(len(alive), 0)
        return (gone, alive)
    sproc1.terminate()
    sproc2.terminate()
    gone, alive = test_2(procs, callback)
    self.assertEqual(set(pids), set([sproc1.pid, sproc2.pid, sproc3.pid]))
    for p in gone:
        self.assertTrue(hasattr(p, 'returncode'))