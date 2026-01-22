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
class TestSystemVirtualMemoryAgainstFree(PsutilTestCase):

    def test_total(self):
        cli_value = free_physmem().total
        psutil_value = psutil.virtual_memory().total
        self.assertEqual(cli_value, psutil_value)

    @retry_on_failure()
    def test_used(self):
        if get_free_version_info() < (3, 3, 12):
            raise self.skipTest('old free version')
        cli_value = free_physmem().used
        psutil_value = psutil.virtual_memory().used
        self.assertAlmostEqual(cli_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_free(self):
        cli_value = free_physmem().free
        psutil_value = psutil.virtual_memory().free
        self.assertAlmostEqual(cli_value, psutil_value, delta=TOLERANCE_SYS_MEM)

    @retry_on_failure()
    def test_shared(self):
        free = free_physmem()
        free_value = free.shared
        if free_value == 0:
            raise unittest.SkipTest("free does not support 'shared' column")
        psutil_value = psutil.virtual_memory().shared
        self.assertAlmostEqual(free_value, psutil_value, delta=TOLERANCE_SYS_MEM, msg='%s %s \n%s' % (free_value, psutil_value, free.output))

    @retry_on_failure()
    def test_available(self):
        out = sh(['free', '-b'])
        lines = out.split('\n')
        if 'available' not in lines[0]:
            raise unittest.SkipTest("free does not support 'available' column")
        else:
            free_value = int(lines[1].split()[-1])
            psutil_value = psutil.virtual_memory().available
            self.assertAlmostEqual(free_value, psutil_value, delta=TOLERANCE_SYS_MEM, msg='%s %s \n%s' % (free_value, psutil_value, out))