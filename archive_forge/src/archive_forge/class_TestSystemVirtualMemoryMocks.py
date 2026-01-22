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
class TestSystemVirtualMemoryMocks(PsutilTestCase):

    def test_warnings_on_misses(self):
        content = textwrap.dedent('            Active(anon):    6145416 kB\n            Active(file):    2950064 kB\n            Inactive(anon):   574764 kB\n            Inactive(file):  1567648 kB\n            MemAvailable:         -1 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            SReclaimable:     346648 kB\n            ').encode()
        with mock_open_content({'/proc/meminfo': content}) as m:
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                ret = psutil.virtual_memory()
                assert m.called
                self.assertEqual(len(ws), 1)
                w = ws[0]
                self.assertIn("memory stats couldn't be determined", str(w.message))
                self.assertIn('cached', str(w.message))
                self.assertIn('shared', str(w.message))
                self.assertIn('active', str(w.message))
                self.assertIn('inactive', str(w.message))
                self.assertIn('buffers', str(w.message))
                self.assertIn('available', str(w.message))
                self.assertEqual(ret.cached, 0)
                self.assertEqual(ret.active, 0)
                self.assertEqual(ret.inactive, 0)
                self.assertEqual(ret.shared, 0)
                self.assertEqual(ret.buffers, 0)
                self.assertEqual(ret.available, 0)
                self.assertEqual(ret.slab, 0)

    @retry_on_failure()
    def test_avail_old_percent(self):
        mems = {}
        with open_binary('/proc/meminfo') as f:
            for line in f:
                fields = line.split()
                mems[fields[0]] = int(fields[1]) * 1024
        a = calculate_avail_vmem(mems)
        if b'MemAvailable:' in mems:
            b = mems[b'MemAvailable:']
            diff_percent = abs(a - b) / a * 100
            self.assertLess(diff_percent, 15)

    def test_avail_old_comes_from_kernel(self):
        content = textwrap.dedent('            Active:          9444728 kB\n            Active(anon):    6145416 kB\n            Active(file):    2950064 kB\n            Buffers:          287952 kB\n            Cached:          4818144 kB\n            Inactive(file):  1578132 kB\n            Inactive(anon):   574764 kB\n            Inactive(file):  1567648 kB\n            MemAvailable:    6574984 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            Shmem:            577588 kB\n            SReclaimable:     346648 kB\n            ').encode()
        with mock_open_content({'/proc/meminfo': content}) as m:
            with warnings.catch_warnings(record=True) as ws:
                ret = psutil.virtual_memory()
            assert m.called
            self.assertEqual(ret.available, 6574984 * 1024)
            w = ws[0]
            self.assertIn("inactive memory stats couldn't be determined", str(w.message))

    def test_avail_old_missing_fields(self):
        content = textwrap.dedent('            Active:          9444728 kB\n            Active(anon):    6145416 kB\n            Buffers:          287952 kB\n            Cached:          4818144 kB\n            Inactive(file):  1578132 kB\n            Inactive(anon):   574764 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            Shmem:            577588 kB\n            ').encode()
        with mock_open_content({'/proc/meminfo': content}) as m:
            with warnings.catch_warnings(record=True) as ws:
                ret = psutil.virtual_memory()
            assert m.called
            self.assertEqual(ret.available, 2057400 * 1024 + 4818144 * 1024)
            w = ws[0]
            self.assertIn("inactive memory stats couldn't be determined", str(w.message))

    def test_avail_old_missing_zoneinfo(self):
        content = textwrap.dedent('            Active:          9444728 kB\n            Active(anon):    6145416 kB\n            Active(file):    2950064 kB\n            Buffers:          287952 kB\n            Cached:          4818144 kB\n            Inactive(file):  1578132 kB\n            Inactive(anon):   574764 kB\n            Inactive(file):  1567648 kB\n            MemFree:         2057400 kB\n            MemTotal:       16325648 kB\n            Shmem:            577588 kB\n            SReclaimable:     346648 kB\n            ').encode()
        with mock_open_content({'/proc/meminfo': content}):
            with mock_open_exception('/proc/zoneinfo', IOError(errno.ENOENT, 'no such file or directory')):
                with warnings.catch_warnings(record=True) as ws:
                    ret = psutil.virtual_memory()
                    self.assertEqual(ret.available, 2057400 * 1024 + 4818144 * 1024)
                    w = ws[0]
                    self.assertIn("inactive memory stats couldn't be determined", str(w.message))

    def test_virtual_memory_mocked(self):
        content = textwrap.dedent('            MemTotal:              100 kB\n            MemFree:               2 kB\n            MemAvailable:          3 kB\n            Buffers:               4 kB\n            Cached:                5 kB\n            SwapCached:            6 kB\n            Active:                7 kB\n            Inactive:              8 kB\n            Active(anon):          9 kB\n            Inactive(anon):        10 kB\n            Active(file):          11 kB\n            Inactive(file):        12 kB\n            Unevictable:           13 kB\n            Mlocked:               14 kB\n            SwapTotal:             15 kB\n            SwapFree:              16 kB\n            Dirty:                 17 kB\n            Writeback:             18 kB\n            AnonPages:             19 kB\n            Mapped:                20 kB\n            Shmem:                 21 kB\n            Slab:                  22 kB\n            SReclaimable:          23 kB\n            SUnreclaim:            24 kB\n            KernelStack:           25 kB\n            PageTables:            26 kB\n            NFS_Unstable:          27 kB\n            Bounce:                28 kB\n            WritebackTmp:          29 kB\n            CommitLimit:           30 kB\n            Committed_AS:          31 kB\n            VmallocTotal:          32 kB\n            VmallocUsed:           33 kB\n            VmallocChunk:          34 kB\n            HardwareCorrupted:     35 kB\n            AnonHugePages:         36 kB\n            ShmemHugePages:        37 kB\n            ShmemPmdMapped:        38 kB\n            CmaTotal:              39 kB\n            CmaFree:               40 kB\n            HugePages_Total:       41 kB\n            HugePages_Free:        42 kB\n            HugePages_Rsvd:        43 kB\n            HugePages_Surp:        44 kB\n            Hugepagesize:          45 kB\n            DirectMap46k:          46 kB\n            DirectMap47M:          47 kB\n            DirectMap48G:          48 kB\n            ').encode()
        with mock_open_content({'/proc/meminfo': content}) as m:
            mem = psutil.virtual_memory()
            assert m.called
            self.assertEqual(mem.total, 100 * 1024)
            self.assertEqual(mem.free, 2 * 1024)
            self.assertEqual(mem.buffers, 4 * 1024)
            self.assertEqual(mem.cached, (5 + 23) * 1024)
            self.assertEqual(mem.shared, 21 * 1024)
            self.assertEqual(mem.active, 7 * 1024)
            self.assertEqual(mem.inactive, 8 * 1024)
            self.assertEqual(mem.slab, 22 * 1024)
            self.assertEqual(mem.available, 3 * 1024)