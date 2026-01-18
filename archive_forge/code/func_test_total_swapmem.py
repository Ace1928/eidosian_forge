import datetime
import errno
import glob
import os
import platform
import re
import signal
import subprocess
import sys
import time
import unittest
import warnings
import psutil
from psutil import WINDOWS
from psutil._compat import FileNotFoundError
from psutil._compat import super
from psutil._compat import which
from psutil.tests import APPVEYOR
from psutil.tests import GITHUB_ACTIONS
from psutil.tests import HAS_BATTERY
from psutil.tests import IS_64BIT
from psutil.tests import PY3
from psutil.tests import PYPY
from psutil.tests import TOLERANCE_DISK_USAGE
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
def test_total_swapmem(self):
    w = wmi.WMI().Win32_PerfRawData_PerfOS_Memory()[0]
    self.assertEqual(int(w.CommitLimit) - psutil.virtual_memory().total, psutil.swap_memory().total)
    if psutil.swap_memory().total == 0:
        self.assertEqual(0, psutil.swap_memory().free)
        self.assertEqual(0, psutil.swap_memory().used)