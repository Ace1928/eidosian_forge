import platform
import re
import time
import unittest
import psutil
from psutil import MACOS
from psutil import POSIX
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
@unittest.skipIf(not MACOS, 'MACOS only')
class TestSystemAPIs(PsutilTestCase):

    @retry_on_failure()
    def test_disks(self):

        def df(path):
            out = sh('df -k "%s"' % path).strip()
            lines = out.split('\n')
            lines.pop(0)
            line = lines.pop(0)
            dev, total, used, free = line.split()[:4]
            if dev == 'none':
                dev = ''
            total = int(total) * 1024
            used = int(used) * 1024
            free = int(free) * 1024
            return (dev, total, used, free)
        for part in psutil.disk_partitions(all=False):
            usage = psutil.disk_usage(part.mountpoint)
            dev, total, used, free = df(part.mountpoint)
            self.assertEqual(part.device, dev)
            self.assertEqual(usage.total, total)
            self.assertAlmostEqual(usage.free, free, delta=TOLERANCE_DISK_USAGE)
            self.assertAlmostEqual(usage.used, used, delta=TOLERANCE_DISK_USAGE)

    def test_cpu_count_logical(self):
        num = sysctl('sysctl hw.logicalcpu')
        self.assertEqual(num, psutil.cpu_count(logical=True))

    def test_cpu_count_cores(self):
        num = sysctl('sysctl hw.physicalcpu')
        self.assertEqual(num, psutil.cpu_count(logical=False))

    @unittest.skipIf(platform.machine() == 'arm64', 'skipped due to #1892')
    def test_cpu_freq(self):
        freq = psutil.cpu_freq()
        self.assertEqual(freq.current * 1000 * 1000, sysctl('sysctl hw.cpufrequency'))
        self.assertEqual(freq.min * 1000 * 1000, sysctl('sysctl hw.cpufrequency_min'))
        self.assertEqual(freq.max * 1000 * 1000, sysctl('sysctl hw.cpufrequency_max'))

    def test_vmem_total(self):
        sysctl_hwphymem = sysctl('sysctl hw.memsize')
        self.assertEqual(sysctl_hwphymem, psutil.virtual_memory().total)

    @retry_on_failure()
    def test_vmem_free(self):
        vmstat_val = vm_stat('free')
        psutil_val = psutil.virtual_memory().free
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_active(self):
        vmstat_val = vm_stat('active')
        psutil_val = psutil.virtual_memory().active
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_inactive(self):
        vmstat_val = vm_stat('inactive')
        psutil_val = psutil.virtual_memory().inactive
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_vmem_wired(self):
        vmstat_val = vm_stat('wired')
        psutil_val = psutil.virtual_memory().wired
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_swapmem_sin(self):
        vmstat_val = vm_stat('Pageins')
        psutil_val = psutil.swap_memory().sin
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_swapmem_sout(self):
        vmstat_val = vm_stat('Pageout')
        psutil_val = psutil.swap_memory().sout
        self.assertAlmostEqual(psutil_val, vmstat_val, delta=TOLERANCE_SYS_MEM)

    def test_net_if_stats(self):
        for name, stats in psutil.net_if_stats().items():
            try:
                out = sh('ifconfig %s' % name)
            except RuntimeError:
                pass
            else:
                self.assertEqual(stats.isup, 'RUNNING' in out, msg=out)
                self.assertEqual(stats.mtu, int(re.findall('mtu (\\d+)', out)[0]))

    @unittest.skipIf(not HAS_BATTERY, 'no battery')
    def test_sensors_battery(self):
        out = sh('pmset -g batt')
        percent = re.search('(\\d+)%', out).group(1)
        drawing_from = re.search("Now drawing from '([^']+)'", out).group(1)
        power_plugged = drawing_from == 'AC Power'
        psutil_result = psutil.sensors_battery()
        self.assertEqual(psutil_result.power_plugged, power_plugged)
        self.assertEqual(psutil_result.percent, int(percent))