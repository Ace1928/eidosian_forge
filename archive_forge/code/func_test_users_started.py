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
@retry_on_failure()
def test_users_started(self):
    out = sh('who -u')
    if not out.strip():
        raise self.skipTest('no users on this system')
    tstamp = None
    started = re.findall('\\d\\d\\d\\d-\\d\\d-\\d\\d \\d\\d:\\d\\d', out)
    if started:
        tstamp = '%Y-%m-%d %H:%M'
    else:
        started = re.findall('[A-Z][a-z][a-z] \\d\\d \\d\\d:\\d\\d', out)
        if started:
            tstamp = '%b %d %H:%M'
        else:
            started = re.findall('[A-Z][a-z][a-z] \\d\\d', out)
            if started:
                tstamp = '%b %d'
            else:
                started = re.findall('[a-z][a-z][a-z] \\d\\d', out)
                if started:
                    tstamp = '%b %d'
                    started = [x.capitalize() for x in started]
    if not tstamp:
        raise unittest.SkipTest('cannot interpret tstamp in who output\n%s' % out)
    with self.subTest(psutil=psutil.users(), who=out):
        for idx, u in enumerate(psutil.users()):
            psutil_value = datetime.datetime.fromtimestamp(u.started).strftime(tstamp)
            self.assertEqual(psutil_value, started[idx])