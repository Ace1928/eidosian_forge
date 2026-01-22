import platform
import signal
import unittest
import psutil
from psutil import AIX
from psutil import FREEBSD
from psutil import LINUX
from psutil import MACOS
from psutil import NETBSD
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil import WINDOWS
from psutil._compat import long
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_CPU_FREQ
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import HAS_SENSORS_FANS
from psutil.tests import HAS_SENSORS_TEMPERATURES
from psutil.tests import PYPY
from psutil.tests import SKIP_SYSCONS
from psutil.tests import PsutilTestCase
from psutil.tests import create_sockets
from psutil.tests import enum
from psutil.tests import is_namedtuple
from psutil.tests import kernel_version
class TestAvailConstantsAPIs(PsutilTestCase):

    def test_PROCFS_PATH(self):
        self.assertEqual(hasattr(psutil, 'PROCFS_PATH'), LINUX or SUNOS or AIX)

    def test_win_priority(self):
        ae = self.assertEqual
        ae(hasattr(psutil, 'ABOVE_NORMAL_PRIORITY_CLASS'), WINDOWS)
        ae(hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'), WINDOWS)
        ae(hasattr(psutil, 'HIGH_PRIORITY_CLASS'), WINDOWS)
        ae(hasattr(psutil, 'IDLE_PRIORITY_CLASS'), WINDOWS)
        ae(hasattr(psutil, 'NORMAL_PRIORITY_CLASS'), WINDOWS)
        ae(hasattr(psutil, 'REALTIME_PRIORITY_CLASS'), WINDOWS)

    def test_linux_ioprio_linux(self):
        ae = self.assertEqual
        ae(hasattr(psutil, 'IOPRIO_CLASS_NONE'), LINUX)
        ae(hasattr(psutil, 'IOPRIO_CLASS_RT'), LINUX)
        ae(hasattr(psutil, 'IOPRIO_CLASS_BE'), LINUX)
        ae(hasattr(psutil, 'IOPRIO_CLASS_IDLE'), LINUX)

    def test_linux_ioprio_windows(self):
        ae = self.assertEqual
        ae(hasattr(psutil, 'IOPRIO_HIGH'), WINDOWS)
        ae(hasattr(psutil, 'IOPRIO_NORMAL'), WINDOWS)
        ae(hasattr(psutil, 'IOPRIO_LOW'), WINDOWS)
        ae(hasattr(psutil, 'IOPRIO_VERYLOW'), WINDOWS)

    @unittest.skipIf(GITHUB_ACTIONS and LINUX, 'unsupported on GITHUB_ACTIONS + LINUX')
    def test_rlimit(self):
        ae = self.assertEqual
        ae(hasattr(psutil, 'RLIM_INFINITY'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_AS'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_CORE'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_CPU'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_DATA'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_FSIZE'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_MEMLOCK'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_NOFILE'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_NPROC'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_RSS'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_STACK'), LINUX or FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_LOCKS'), LINUX)
        if POSIX:
            if kernel_version() >= (2, 6, 8):
                ae(hasattr(psutil, 'RLIMIT_MSGQUEUE'), LINUX)
            if kernel_version() >= (2, 6, 12):
                ae(hasattr(psutil, 'RLIMIT_NICE'), LINUX)
            if kernel_version() >= (2, 6, 12):
                ae(hasattr(psutil, 'RLIMIT_RTPRIO'), LINUX)
            if kernel_version() >= (2, 6, 25):
                ae(hasattr(psutil, 'RLIMIT_RTTIME'), LINUX)
            if kernel_version() >= (2, 6, 8):
                ae(hasattr(psutil, 'RLIMIT_SIGPENDING'), LINUX)
        ae(hasattr(psutil, 'RLIMIT_SWAP'), FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_SBSIZE'), FREEBSD)
        ae(hasattr(psutil, 'RLIMIT_NPTS'), FREEBSD)