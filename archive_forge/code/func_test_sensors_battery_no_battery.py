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
@unittest.skipIf(HAS_BATTERY, 'has battery')
def test_sensors_battery_no_battery(self):
    with self.assertRaises(RuntimeError):
        sysctl('hw.acpi.battery.life')
        sysctl('hw.acpi.battery.time')
        sysctl('hw.acpi.acline')
    self.assertIsNone(psutil.sensors_battery())