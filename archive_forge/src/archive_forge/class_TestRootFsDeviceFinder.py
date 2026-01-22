from __future__ import division
import collections
import contextlib
import errno
import glob
import io
import os
import re
import shutil
import socket
import struct
import textwrap
import time
import unittest
import warnings
import psutil
from psutil import LINUX
from psutil._compat import PY3
from psutil._compat import FileNotFoundError
from psutil._compat import basestring
from psutil._compat import u
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import GLOBAL_TIMEOUT
from psutil.tests import HAS_BATTERY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_GETLOADAVG
from psutil.tests import HAS_RLIMIT
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import ThreadTask
from psutil.tests import call_until
from psutil.tests import mock
from psutil.tests import reload_module
from psutil.tests import retry_on_failure
from psutil.tests import safe_rmpath
from psutil.tests import sh
from psutil.tests import skip_on_not_implemented
from psutil.tests import which
@unittest.skipIf(not LINUX, 'LINUX only')
class TestRootFsDeviceFinder(PsutilTestCase):

    def setUp(self):
        dev = os.stat('/').st_dev
        self.major = os.major(dev)
        self.minor = os.minor(dev)

    def test_call_methods(self):
        finder = RootFsDeviceFinder()
        if os.path.exists('/proc/partitions'):
            finder.ask_proc_partitions()
        else:
            self.assertRaises(FileNotFoundError, finder.ask_proc_partitions)
        if os.path.exists('/sys/dev/block/%s:%s/uevent' % (self.major, self.minor)):
            finder.ask_sys_dev_block()
        else:
            self.assertRaises(FileNotFoundError, finder.ask_sys_dev_block)
        finder.ask_sys_class_block()

    @unittest.skipIf(GITHUB_ACTIONS, 'unsupported on GITHUB_ACTIONS')
    def test_comparisons(self):
        finder = RootFsDeviceFinder()
        self.assertIsNotNone(finder.find())
        a = b = c = None
        if os.path.exists('/proc/partitions'):
            a = finder.ask_proc_partitions()
        if os.path.exists('/sys/dev/block/%s:%s/uevent' % (self.major, self.minor)):
            b = finder.ask_sys_class_block()
        c = finder.ask_sys_dev_block()
        base = a or b or c
        if base and a:
            self.assertEqual(base, a)
        if base and b:
            self.assertEqual(base, b)
        if base and c:
            self.assertEqual(base, c)

    @unittest.skipIf(not which('findmnt'), 'findmnt utility not available')
    @unittest.skipIf(GITHUB_ACTIONS, 'unsupported on GITHUB_ACTIONS')
    def test_against_findmnt(self):
        psutil_value = RootFsDeviceFinder().find()
        findmnt_value = sh('findmnt -o SOURCE -rn /')
        self.assertEqual(psutil_value, findmnt_value)

    def test_disk_partitions_mocked(self):
        with mock.patch('psutil._pslinux.cext.disk_partitions', return_value=[('/dev/root', '/', 'ext4', 'rw')]) as m:
            part = psutil.disk_partitions()[0]
            assert m.called
            if not GITHUB_ACTIONS:
                self.assertNotEqual(part.device, '/dev/root')
                self.assertEqual(part.device, RootFsDeviceFinder().find())
            else:
                self.assertEqual(part.device, '/dev/root')