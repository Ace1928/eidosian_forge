import datetime
import errno
import os
import re
import subprocess
import time
import unittest
import psutil
from psutil import AIX
from psutil import BSD
from psutil import LINUX
from psutil import MACOS
from psutil import OPENBSD
from psutil import POSIX
from psutil import SUNOS
from psutil.tests import HAS_NET_IO_COUNTERS
from psutil.tests import PYTHON_EXE
from psutil.tests import PsutilTestCase
from psutil.tests import mock
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import skip_on_access_denied
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
def test_pid_exists_let_raise(self):
    with mock.patch('psutil._psposix.os.kill', side_effect=OSError(errno.EBADF, '')) as m:
        self.assertRaises(OSError, psutil._psposix.pid_exists, os.getpid())
        assert m.called