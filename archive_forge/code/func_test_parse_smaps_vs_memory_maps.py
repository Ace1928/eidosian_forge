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
@retry_on_failure()
def test_parse_smaps_vs_memory_maps(self):
    sproc = self.spawn_testproc()
    uss, pss, swap = psutil._pslinux.Process(sproc.pid)._parse_smaps()
    maps = psutil.Process(sproc.pid).memory_maps(grouped=False)
    self.assertAlmostEqual(uss, sum([x.private_dirty + x.private_clean for x in maps]), delta=4096)
    self.assertAlmostEqual(pss, sum([x.pss for x in maps]), delta=4096)
    self.assertAlmostEqual(swap, sum([x.swap for x in maps]), delta=4096)