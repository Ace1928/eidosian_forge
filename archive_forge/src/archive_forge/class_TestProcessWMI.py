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
class TestProcessWMI(WindowsTestCase):
    """Compare Process API results with WMI."""

    @classmethod
    def setUpClass(cls):
        cls.pid = spawn_testproc().pid

    @classmethod
    def tearDownClass(cls):
        terminate(cls.pid)

    def test_name(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        self.assertEqual(p.name(), w.Caption)

    @unittest.skipIf(GITHUB_ACTIONS, 'unreliable path on GITHUB_ACTIONS')
    def test_exe(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        self.assertEqual(p.exe().lower(), w.ExecutablePath.lower())

    def test_cmdline(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        self.assertEqual(' '.join(p.cmdline()), w.CommandLine.replace('"', ''))

    def test_username(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        domain, _, username = w.GetOwner()
        username = '%s\\%s' % (domain, username)
        self.assertEqual(p.username(), username)

    @retry_on_failure()
    def test_memory_rss(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        rss = p.memory_info().rss
        self.assertEqual(rss, int(w.WorkingSetSize))

    @retry_on_failure()
    def test_memory_vms(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        vms = p.memory_info().vms
        wmi_usage = int(w.PageFileUsage)
        if vms not in (wmi_usage, wmi_usage * 1024):
            raise self.fail('wmi=%s, psutil=%s' % (wmi_usage, vms))

    def test_create_time(self):
        w = wmi.WMI().Win32_Process(ProcessId=self.pid)[0]
        p = psutil.Process(self.pid)
        wmic_create = str(w.CreationDate.split('.')[0])
        psutil_create = time.strftime('%Y%m%d%H%M%S', time.localtime(p.create_time()))
        self.assertEqual(wmic_create, psutil_create)