from __future__ import print_function
import functools
import os
import platform
import unittest
import psutil
import psutil._common
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import ProcessLookupError
from psutil._compat import super
from psutil.tests import HAS_CPU_AFFINITY
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_ENVIRON
from psutil.tests import HAS_IONICE
from psutil.tests import HAS_MEMORY_MAPS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_PROC_CPU_NUM
from psutil.tests import HAS_PROC_IO_COUNTERS
from psutil.tests import HAS_RLIMIT
from psutil.tests import HAS_SENSORS_BATTERY
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import TestMemoryLeak
from psutil.tests import create_sockets
from psutil.tests import get_testfn
from psutil.tests import process_namespace
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import system_namespace
from psutil.tests import terminate
class TestModuleFunctionsLeaks(TestMemoryLeak):
    """Test leaks of psutil module functions."""

    def test_coverage(self):
        ns = system_namespace()
        ns.test_class_coverage(self, ns.all)

    @fewtimes_if_linux()
    def test_cpu_count(self):
        self.execute(lambda: psutil.cpu_count(logical=True))

    @fewtimes_if_linux()
    def test_cpu_count_cores(self):
        self.execute(lambda: psutil.cpu_count(logical=False))

    @fewtimes_if_linux()
    def test_cpu_times(self):
        self.execute(psutil.cpu_times)

    @fewtimes_if_linux()
    def test_per_cpu_times(self):
        self.execute(lambda: psutil.cpu_times(percpu=True))

    @fewtimes_if_linux()
    def test_cpu_stats(self):
        self.execute(psutil.cpu_stats)

    @fewtimes_if_linux()
    @unittest.skipIf(MACOS and platform.machine() == 'arm64', 'skipped due to #1892')
    @unittest.skipIf(not HAS_CPU_FREQ, 'not supported')
    def test_cpu_freq(self):
        self.execute(psutil.cpu_freq)

    @unittest.skipIf(not WINDOWS, 'WINDOWS only')
    def test_getloadavg(self):
        psutil.getloadavg()
        self.execute(psutil.getloadavg)

    def test_virtual_memory(self):
        self.execute(psutil.virtual_memory)

    @unittest.skipIf(SUNOS, 'worthless on SUNOS (uses a subprocess)')
    def test_swap_memory(self):
        self.execute(psutil.swap_memory)

    def test_pid_exists(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.pid_exists(os.getpid()), times=times)

    def test_disk_usage(self):
        times = FEW_TIMES if POSIX else self.times
        self.execute(lambda: psutil.disk_usage('.'), times=times)

    def test_disk_partitions(self):
        self.execute(psutil.disk_partitions)

    @unittest.skipIf(LINUX and (not os.path.exists('/proc/diskstats')), '/proc/diskstats not available on this Linux version')
    @fewtimes_if_linux()
    def test_disk_io_counters(self):
        self.execute(lambda: psutil.disk_io_counters(nowrap=False))

    @fewtimes_if_linux()
    def test_pids(self):
        self.execute(psutil.pids)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_NET_IO_COUNTERS, 'not supported')
    def test_net_io_counters(self):
        self.execute(lambda: psutil.net_io_counters(nowrap=False))

    @fewtimes_if_linux()
    @unittest.skipIf(MACOS and os.getuid() != 0, 'need root access')
    def test_net_connections(self):
        psutil.net_connections(kind='all')
        with create_sockets():
            self.execute(lambda: psutil.net_connections(kind='all'))

    def test_net_if_addrs(self):
        tolerance = 80 * 1024 if WINDOWS else self.tolerance
        self.execute(psutil.net_if_addrs, tolerance=tolerance)

    def test_net_if_stats(self):
        self.execute(psutil.net_if_stats)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_BATTERY, 'not supported')
    def test_sensors_battery(self):
        self.execute(psutil.sensors_battery)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_TEMPERATURES, 'not supported')
    def test_sensors_temperatures(self):
        self.execute(psutil.sensors_temperatures)

    @fewtimes_if_linux()
    @unittest.skipIf(not HAS_SENSORS_FANS, 'not supported')
    def test_sensors_fans(self):
        self.execute(psutil.sensors_fans)

    @fewtimes_if_linux()
    def test_boot_time(self):
        self.execute(psutil.boot_time)

    def test_users(self):
        self.execute(psutil.users)

    def test_set_debug(self):
        self.execute(lambda: psutil._set_debug(False))
    if WINDOWS:

        def test_win_service_iter(self):
            self.execute(cext.winservice_enumerate)

        def test_win_service_get(self):
            pass

        def test_win_service_get_config(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_config(name))

        def test_win_service_get_status(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_status(name))

        def test_win_service_get_description(self):
            name = next(psutil.win_service_iter()).name()
            self.execute(lambda: cext.winservice_query_descr(name))