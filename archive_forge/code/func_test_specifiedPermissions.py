from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_specifiedPermissions(self) -> None:
    """
        Test specifying the permissions used on the log file.
        """
    log1 = logfile.LogFile(self.name, self.dir, defaultMode=54)
    self.addCleanup(log1.close)
    mode = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
    if runtime.platform.isWindows():
        self.assertEqual(mode, 292)
    else:
        self.assertEqual(mode, 54)