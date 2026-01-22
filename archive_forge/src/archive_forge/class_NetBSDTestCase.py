import datetime
import os
import re
import time
import unittest
import psutil
from psutil import BSD
from psutil import FREEBSD
from psutil import NETBSD
from psutil import OPENBSD
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
@unittest.skipIf(not NETBSD, 'NETBSD only')
class NetBSDTestCase(PsutilTestCase):

    @staticmethod
    def parse_meminfo(look_for):
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith(look_for):
                    return int(line.split()[1]) * 1024
        raise ValueError("can't find %s" % look_for)

    def test_vmem_total(self):
        self.assertEqual(psutil.virtual_memory().total, self.parse_meminfo('MemTotal:'))

    def test_vmem_free(self):
        self.assertAlmostEqual(psutil.virtual_memory().free, self.parse_meminfo('MemFree:'), delta=TOLERANCE_SYS_MEM)

    def test_vmem_buffers(self):
        self.assertAlmostEqual(psutil.virtual_memory().buffers, self.parse_meminfo('Buffers:'), delta=TOLERANCE_SYS_MEM)

    def test_vmem_shared(self):
        self.assertAlmostEqual(psutil.virtual_memory().shared, self.parse_meminfo('MemShared:'), delta=TOLERANCE_SYS_MEM)

    def test_vmem_cached(self):
        self.assertAlmostEqual(psutil.virtual_memory().cached, self.parse_meminfo('Cached:'), delta=TOLERANCE_SYS_MEM)

    def test_swapmem_total(self):
        self.assertAlmostEqual(psutil.swap_memory().total, self.parse_meminfo('SwapTotal:'), delta=TOLERANCE_SYS_MEM)

    def test_swapmem_free(self):
        self.assertAlmostEqual(psutil.swap_memory().free, self.parse_meminfo('SwapFree:'), delta=TOLERANCE_SYS_MEM)

    def test_swapmem_used(self):
        smem = psutil.swap_memory()
        self.assertEqual(smem.used, smem.total - smem.free)

    def test_cpu_stats_interrupts(self):
        with open('/proc/stat', 'rb') as f:
            for line in f:
                if line.startswith(b'intr'):
                    interrupts = int(line.split()[1])
                    break
            else:
                raise ValueError("couldn't find line")
        self.assertAlmostEqual(psutil.cpu_stats().interrupts, interrupts, delta=1000)

    def test_cpu_stats_ctx_switches(self):
        with open('/proc/stat', 'rb') as f:
            for line in f:
                if line.startswith(b'ctxt'):
                    ctx_switches = int(line.split()[1])
                    break
            else:
                raise ValueError("couldn't find line")
        self.assertAlmostEqual(psutil.cpu_stats().ctx_switches, ctx_switches, delta=1000)