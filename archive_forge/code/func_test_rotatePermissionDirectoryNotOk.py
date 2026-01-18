from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
@skipIf(runtime.platform.isWindows(), 'Making read-only directories on Windows is too complex for this test to reasonably do.')
def test_rotatePermissionDirectoryNotOk(self) -> None:
    """
        L{DailyLogFile.rotate} doesn't do anything if the directory containing
        the log files can't be written to.
        """
    log = logfile.DailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    os.chmod(log.directory, 292)
    self.addCleanup(os.chmod, log.directory, 493)
    previousFile = log._file
    log.rotate()
    self.assertEqual(previousFile, log._file)