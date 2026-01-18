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
@unittest.skipIf(not HAS_BATTERY, 'no battery')
def test_sensors_battery_against_sysctl(self):
    self.assertEqual(psutil.sensors_battery().percent, sysctl('hw.acpi.battery.life'))
    self.assertEqual(psutil.sensors_battery().power_plugged, sysctl('hw.acpi.acline') == 1)
    secsleft = psutil.sensors_battery().secsleft
    if secsleft < 0:
        self.assertEqual(sysctl('hw.acpi.battery.time'), -1)
    else:
        self.assertEqual(secsleft, sysctl('hw.acpi.battery.time') * 60)