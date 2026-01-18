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
def test_special_pid(self):
    p = psutil.Process(4)
    self.assertEqual(p.name(), 'System')
    str(p)
    p.username()
    self.assertGreaterEqual(p.create_time(), 0.0)
    try:
        rss, vms = p.memory_info()[:2]
    except psutil.AccessDenied:
        if platform.uname()[1] not in ('vista', 'win-7', 'win7'):
            raise
    else:
        self.assertGreater(rss, 0)