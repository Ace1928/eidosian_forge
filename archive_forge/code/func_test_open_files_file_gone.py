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
def test_open_files_file_gone(self):
    p = psutil.Process()
    files = p.open_files()
    with open(self.get_testfn(), 'w'):
        call_until(p.open_files, 'len(ret) != %i' % len(files))
        with mock.patch('psutil._pslinux.os.readlink', side_effect=OSError(errno.ENOENT, '')) as m:
            self.assertEqual(p.open_files(), [])
            assert m.called
        with mock.patch('psutil._pslinux.os.readlink', side_effect=OSError(errno.EINVAL, '')) as m:
            self.assertEqual(p.open_files(), [])
            assert m.called