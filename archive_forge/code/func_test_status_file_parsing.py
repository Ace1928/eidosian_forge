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
def test_status_file_parsing(self):
    content = textwrap.dedent('            Uid:\t1000\t1001\t1002\t1003\n            Gid:\t1004\t1005\t1006\t1007\n            Threads:\t66\n            Cpus_allowed:\tf\n            Cpus_allowed_list:\t0-7\n            voluntary_ctxt_switches:\t12\n            nonvoluntary_ctxt_switches:\t13').encode()
    with mock_open_content({'/proc/%s/status' % os.getpid(): content}):
        p = psutil.Process()
        self.assertEqual(p.num_ctx_switches().voluntary, 12)
        self.assertEqual(p.num_ctx_switches().involuntary, 13)
        self.assertEqual(p.num_threads(), 66)
        uids = p.uids()
        self.assertEqual(uids.real, 1000)
        self.assertEqual(uids.effective, 1001)
        self.assertEqual(uids.saved, 1002)
        gids = p.gids()
        self.assertEqual(gids.real, 1004)
        self.assertEqual(gids.effective, 1005)
        self.assertEqual(gids.saved, 1006)
        self.assertEqual(p._proc._get_eligible_cpus(), list(range(8)))