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