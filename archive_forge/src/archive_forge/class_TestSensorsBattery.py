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
@unittest.skipIf(not HAS_BATTERY, 'no battery')
class TestSensorsBattery(PsutilTestCase):

    @unittest.skipIf(not which('acpi'), 'acpi utility not available')
    def test_percent(self):
        out = sh('acpi -b')
        acpi_value = int(out.split(',')[1].strip().replace('%', ''))
        psutil_value = psutil.sensors_battery().percent
        self.assertAlmostEqual(acpi_value, psutil_value, delta=1)

    def test_emulate_power_plugged(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith(('AC0/online', 'AC/online')):
                return io.BytesIO(b'1')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock) as m:
            self.assertEqual(psutil.sensors_battery().power_plugged, True)
            self.assertEqual(psutil.sensors_battery().secsleft, psutil.POWER_TIME_UNLIMITED)
            assert m.called

    def test_emulate_power_plugged_2(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith(('AC0/online', 'AC/online')):
                raise IOError(errno.ENOENT, '')
            elif name.endswith('/status'):
                return io.StringIO(u('charging'))
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock) as m:
            self.assertEqual(psutil.sensors_battery().power_plugged, True)
            assert m.called

    def test_emulate_power_not_plugged(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith(('AC0/online', 'AC/online')):
                return io.BytesIO(b'0')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock) as m:
            self.assertEqual(psutil.sensors_battery().power_plugged, False)
            assert m.called

    def test_emulate_power_not_plugged_2(self):

        def open_mock(name, *args, **kwargs):
            if name.endswith(('AC0/online', 'AC/online')):
                raise IOError(errno.ENOENT, '')
            elif name.endswith('/status'):
                return io.StringIO(u('discharging'))
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock) as m:
            self.assertEqual(psutil.sensors_battery().power_plugged, False)
            assert m.called

    def test_emulate_power_undetermined(self):

        def open_mock(name, *args, **kwargs):
            if name.startswith(('/sys/class/power_supply/AC0/online', '/sys/class/power_supply/AC/online')):
                raise IOError(errno.ENOENT, '')
            elif name.startswith('/sys/class/power_supply/BAT0/status'):
                return io.BytesIO(b'???')
            else:
                return orig_open(name, *args, **kwargs)
        orig_open = open
        patch_point = 'builtins.open' if PY3 else '__builtin__.open'
        with mock.patch(patch_point, side_effect=open_mock) as m:
            self.assertIsNone(psutil.sensors_battery().power_plugged)
            assert m.called

    def test_emulate_energy_full_0(self):
        with mock_open_content({'/sys/class/power_supply/BAT0/energy_full': b'0'}) as m:
            self.assertEqual(psutil.sensors_battery().percent, 0)
            assert m.called

    def test_emulate_energy_full_not_avail(self):
        with mock_open_exception('/sys/class/power_supply/BAT0/energy_full', IOError(errno.ENOENT, '')):
            with mock_open_exception('/sys/class/power_supply/BAT0/charge_full', IOError(errno.ENOENT, '')):
                with mock_open_content({'/sys/class/power_supply/BAT0/capacity': b'88'}):
                    self.assertEqual(psutil.sensors_battery().percent, 88)

    def test_emulate_no_power(self):
        with mock_open_exception('/sys/class/power_supply/AC/online', IOError(errno.ENOENT, '')):
            with mock_open_exception('/sys/class/power_supply/AC0/online', IOError(errno.ENOENT, '')):
                with mock_open_exception('/sys/class/power_supply/BAT0/status', IOError(errno.ENOENT, '')):
                    self.assertIsNone(psutil.sensors_battery().power_plugged)