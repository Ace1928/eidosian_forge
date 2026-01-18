import datetime
import os
import re
import time
import unittest
import psutil
from psutil import BSD
from psutil import FREEBSD
from psutil import NETBSD
from psutil import OPENBSD
from psutil.tests import HAS_BATTERY
from psutil.tests import TOLERANCE_SYS_MEM
from psutil.tests import PsutilTestCase
from psutil.tests import retry_on_failure
from psutil.tests import sh
from psutil.tests import spawn_testproc
from psutil.tests import terminate
from psutil.tests import which
def test_uids_gids(self):
    out = sh('procstat -s %s' % self.pid)
    euid, ruid, suid, egid, rgid, sgid = out.split('\n')[1].split()[2:8]
    p = psutil.Process(self.pid)
    uids = p.uids()
    gids = p.gids()
    self.assertEqual(uids.real, int(ruid))
    self.assertEqual(uids.effective, int(euid))
    self.assertEqual(uids.saved, int(suid))
    self.assertEqual(gids.real, int(rgid))
    self.assertEqual(gids.effective, int(egid))
    self.assertEqual(gids.saved, int(sgid))