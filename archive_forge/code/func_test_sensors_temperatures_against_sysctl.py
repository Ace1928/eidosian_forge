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
def test_sensors_temperatures_against_sysctl(self):
    num_cpus = psutil.cpu_count(True)
    for cpu in range(num_cpus):
        sensor = 'dev.cpu.%s.temperature' % cpu
        try:
            sysctl_result = int(float(sysctl(sensor)[:-1]))
        except RuntimeError:
            self.skipTest('temperatures not supported by kernel')
        self.assertAlmostEqual(psutil.sensors_temperatures()['coretemp'][cpu].current, sysctl_result, delta=10)
        sensor = 'dev.cpu.%s.coretemp.tjmax' % cpu
        sysctl_result = int(float(sysctl(sensor)[:-1]))
        self.assertEqual(psutil.sensors_temperatures()['coretemp'][cpu].high, sysctl_result)