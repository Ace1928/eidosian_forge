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
class TestSystemNetIOCounters(PsutilTestCase):

    @unittest.skipIf(not which('ifconfig'), 'ifconfig utility not available')
    @retry_on_failure()
    def test_against_ifconfig(self):

        def ifconfig(nic):
            ret = {}
            out = sh('ifconfig %s' % nic)
            ret['packets_recv'] = int(re.findall('RX packets[: ](\\d+)', out)[0])
            ret['packets_sent'] = int(re.findall('TX packets[: ](\\d+)', out)[0])
            ret['errin'] = int(re.findall('errors[: ](\\d+)', out)[0])
            ret['errout'] = int(re.findall('errors[: ](\\d+)', out)[1])
            ret['dropin'] = int(re.findall('dropped[: ](\\d+)', out)[0])
            ret['dropout'] = int(re.findall('dropped[: ](\\d+)', out)[1])
            ret['bytes_recv'] = int(re.findall('RX (?:packets \\d+ +)?bytes[: ](\\d+)', out)[0])
            ret['bytes_sent'] = int(re.findall('TX (?:packets \\d+ +)?bytes[: ](\\d+)', out)[0])
            return ret
        nio = psutil.net_io_counters(pernic=True, nowrap=False)
        for name, stats in nio.items():
            try:
                ifconfig_ret = ifconfig(name)
            except RuntimeError:
                continue
            self.assertAlmostEqual(stats.bytes_recv, ifconfig_ret['bytes_recv'], delta=1024 * 5)
            self.assertAlmostEqual(stats.bytes_sent, ifconfig_ret['bytes_sent'], delta=1024 * 5)
            self.assertAlmostEqual(stats.packets_recv, ifconfig_ret['packets_recv'], delta=1024)
            self.assertAlmostEqual(stats.packets_sent, ifconfig_ret['packets_sent'], delta=1024)
            self.assertAlmostEqual(stats.errin, ifconfig_ret['errin'], delta=10)
            self.assertAlmostEqual(stats.errout, ifconfig_ret['errout'], delta=10)
            self.assertAlmostEqual(stats.dropin, ifconfig_ret['dropin'], delta=10)
            self.assertAlmostEqual(stats.dropout, ifconfig_ret['dropout'], delta=10)