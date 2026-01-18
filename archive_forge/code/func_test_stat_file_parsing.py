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
def test_stat_file_parsing(self):
    args = ['0', '(cat)', 'Z', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '2', '3', '4', '5', '0', '0', '0', '0', '6', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '6', '0', '0', '7']
    content = ' '.join(args).encode()
    with mock_open_content({'/proc/%s/stat' % os.getpid(): content}):
        p = psutil.Process()
        self.assertEqual(p.name(), 'cat')
        self.assertEqual(p.status(), psutil.STATUS_ZOMBIE)
        self.assertEqual(p.ppid(), 1)
        self.assertEqual(p.create_time(), 6 / CLOCK_TICKS + psutil.boot_time())
        cpu = p.cpu_times()
        self.assertEqual(cpu.user, 2 / CLOCK_TICKS)
        self.assertEqual(cpu.system, 3 / CLOCK_TICKS)
        self.assertEqual(cpu.children_user, 4 / CLOCK_TICKS)
        self.assertEqual(cpu.children_system, 5 / CLOCK_TICKS)
        self.assertEqual(cpu.iowait, 7 / CLOCK_TICKS)
        self.assertEqual(p.cpu_num(), 6)