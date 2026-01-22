from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
class DailyLogFileTests(TestCase):
    """
    Test rotating log file.
    """

    def setUp(self) -> None:
        self.dir = self.mktemp()
        os.makedirs(self.dir)
        self.name = 'testdaily.log'
        self.path = os.path.join(self.dir, self.name)

    def test_writing(self) -> None:
        """
        A daily log file can be written to like an ordinary log file.
        """
        with contextlib.closing(RiggedDailyLogFile(self.name, self.dir)) as log:
            log.write('123')
            log.write('456')
            log.flush()
            log.write('7890')
        with open(self.path) as f:
            self.assertEqual(f.read(), '1234567890')

    def test_rotation(self) -> None:
        """
        Daily log files rotate daily.
        """
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        days = [self.path + '.' + log.suffix(day * 86400) for day in range(3)]
        log._clock = 0.0
        log.write('123')
        log._clock = 43200
        log.write('4567890')
        log._clock = 86400
        log.write('1' * 11)
        self.assertTrue(os.path.exists(days[0]))
        self.assertFalse(os.path.exists(days[1]))
        log._clock = 172800
        log.write('')
        self.assertTrue(os.path.exists(days[0]))
        self.assertTrue(os.path.exists(days[1]))
        self.assertFalse(os.path.exists(days[2]))
        log._clock = 259199
        log.write('3')
        self.assertFalse(os.path.exists(days[2]))

    def test_getLog(self) -> None:
        """
        Test retrieving log files with L{DailyLogFile.getLog}.
        """
        data = ['1\n', '2\n', '3\n']
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        for d in data:
            log.write(d)
        log.flush()
        r = log.getLog(0.0)
        self.addCleanup(r.close)
        self.assertEqual(data, r.readLines())
        self.assertRaises(ValueError, log.getLog, 86400)
        log._clock = 86401
        r.close()
        log.rotate()
        r = log.getLog(0)
        self.addCleanup(r.close)
        self.assertEqual(data, r.readLines())

    def test_rotateAlreadyExists(self) -> None:
        """
        L{DailyLogFile.rotate} doesn't do anything if they new log file already
        exists on the disk.
        """
        log = RiggedDailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        newFilePath = f'{log.path}.{log.suffix(log.lastDate)}'
        with open(newFilePath, 'w') as fp:
            fp.write('123')
        previousFile = log._file
        log.rotate()
        self.assertEqual(previousFile, log._file)

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

    def test_toDate(self) -> None:
        """
        Test that L{DailyLogFile.toDate} converts its timestamp argument to a
        time tuple (year, month, day).
        """
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        timestamp = time.mktime((2000, 1, 1, 0, 0, 0, 0, 0, 0))
        self.assertEqual((2000, 1, 1), log.toDate(timestamp))

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

    def test_toDateUsesArgumentsToMakeADate(self) -> None:
        """
        Test that L{DailyLogFile.toDate} uses its arguments to create a new
        date.
        """
        log = logfile.DailyLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        date = (2014, 10, 22)
        seconds = time.mktime(date + (0,) * 6)
        logDate = log.toDate(seconds)
        self.assertEqual(date, logDate)