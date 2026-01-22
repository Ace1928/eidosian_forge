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
class TestDualProcessImplementation(PsutilTestCase):
    """Certain APIs on Windows have 2 internal implementations, one
    based on documented Windows APIs, another one based
    NtQuerySystemInformation() which gets called as fallback in
    case the first fails because of limited permission error.
    Here we test that the two methods return the exact same value,
    see:
    https://github.com/giampaolo/psutil/issues/304.
    """

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_memory_info(self):
        mem_1 = psutil.Process(self.pid).memory_info()
        with mock.patch('psutil._psplatform.cext.proc_memory_info', side_effect=OSError(errno.EPERM, 'msg')) as fun:
            mem_2 = psutil.Process(self.pid).memory_info()
            self.assertEqual(len(mem_1), len(mem_2))
            for i in range(len(mem_1)):
                self.assertGreaterEqual(mem_1[i], 0)
                self.assertGreaterEqual(mem_2[i], 0)
                self.assertAlmostEqual(mem_1[i], mem_2[i], delta=512)
            assert fun.called

    def test_create_time(self):
        ctime = psutil.Process(self.pid).create_time()
        with mock.patch('psutil._psplatform.cext.proc_times', side_effect=OSError(errno.EPERM, 'msg')) as fun:
            self.assertEqual(psutil.Process(self.pid).create_time(), ctime)
            assert fun.called

    def test_cpu_times(self):
        cpu_times_1 = psutil.Process(self.pid).cpu_times()
        with mock.patch('psutil._psplatform.cext.proc_times', side_effect=OSError(errno.EPERM, 'msg')) as fun:
            cpu_times_2 = psutil.Process(self.pid).cpu_times()
            assert fun.called
            self.assertAlmostEqual(cpu_times_1.user, cpu_times_2.user, delta=0.01)
            self.assertAlmostEqual(cpu_times_1.system, cpu_times_2.system, delta=0.01)

    def test_io_counters(self):
        io_counters_1 = psutil.Process(self.pid).io_counters()
        with mock.patch('psutil._psplatform.cext.proc_io_counters', side_effect=OSError(errno.EPERM, 'msg')) as fun:
            io_counters_2 = psutil.Process(self.pid).io_counters()
            for i in range(len(io_counters_1)):
                self.assertAlmostEqual(io_counters_1[i], io_counters_2[i], delta=5)
            assert fun.called

    def test_num_handles(self):
        num_handles = psutil.Process(self.pid).num_handles()
        with mock.patch('psutil._psplatform.cext.proc_num_handles', side_effect=OSError(errno.EPERM, 'msg')) as fun:
            self.assertEqual(psutil.Process(self.pid).num_handles(), num_handles)
            assert fun.called

    def test_cmdline(self):
        for pid in psutil.pids():
            try:
                a = cext.proc_cmdline(pid, use_peb=True)
                b = cext.proc_cmdline(pid, use_peb=False)
            except OSError as err:
                err = convert_oserror(err)
                if not isinstance(err, (psutil.AccessDenied, psutil.NoSuchProcess)):
                    raise
            else:
                self.assertEqual(a, b)