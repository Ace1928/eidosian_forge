from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_rotatePermissionFileNotOk(self) -> None:
    """
        L{DailyLogFile.rotate} doesn't do anything if the log file can't be
        written to.
        """
    log = logfile.DailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    os.chmod(log.path, 292)
    previousFile = log._file
    log.rotate()
    self.assertEqual(previousFile, log._file)