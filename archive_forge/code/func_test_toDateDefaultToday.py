from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
def test_toDateDefaultToday(self) -> None:
    """
        Test that L{DailyLogFile.toDate} returns today's date by default.

        By mocking L{time.localtime}, we ensure that L{DailyLogFile.toDate}
        returns the first 3 values of L{time.localtime} which is the current
        date.

        Note that we don't compare the *real* result of L{DailyLogFile.toDate}
        to the *real* current date, as there's a slight possibility that the
        date changes between the 2 function calls.
        """

    def mock_localtime(*args: object) -> list[int]:
        self.assertEqual((), args)
        return list(range(0, 9))
    log = logfile.DailyLogFile(self.name, self.dir)
    self.addCleanup(log.close)
    self.patch(time, 'localtime', mock_localtime)
    logDate = log.toDate()
    self.assertEqual([0, 1, 2], logDate)