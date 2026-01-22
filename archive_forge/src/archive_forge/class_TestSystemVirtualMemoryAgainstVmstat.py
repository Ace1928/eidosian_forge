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
class TestSystemVirtualMemoryAgainstVmstat(PsutilTestCase):

    def test_total(self):
        vmstat_value = vmstat('total memory') * 1024
        psutil_value = psutil.virtual_memory().total
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_used(self):
        if get_free_version_info() < (3, 3, 12):
            raise self.skipTest('old free version')
        vmstat_value = vmstat('used memory') * 1024
        psutil_value = psutil.virtual_memory().used
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_free(self):
        vmstat_value = vmstat('free memory') * 1024
        psutil_value = psutil.virtual_memory().free
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_buffers(self):
        vmstat_value = vmstat('buffer memory') * 1024
        psutil_value = psutil.virtual_memory().buffers
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_active(self):
        vmstat_value = vmstat('active memory') * 1024
        psutil_value = psutil.virtual_memory().active
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_inactive(self):
        vmstat_value = vmstat('inactive memory') * 1024
        psutil_value = psutil.virtual_memory().inactive
        self.assertAlmostEqual(vmstat_value, psutil_value, delta=TOLERANCE_SYS_MEM)