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
def test_vmem_total(self):
    sysctl_hwphymem = sysctl('sysctl hw.memsize')
    self.assertEqual(sysctl_hwphymem, psutil.virtual_memory().total)