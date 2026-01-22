from __future__ import annotations
import contextlib
import errno
import os
import stat
import time
from unittest import skipIf
from twisted.python import logfile, runtime
from twisted.trial.unittest import TestCase
class LogFileTests(TestCase):
    """
    Test the rotating log file.
    """

    def setUp(self) -> None:
        self.dir = self.mktemp()
        os.makedirs(self.dir)
        self.name = 'test.log'
        self.path = os.path.join(self.dir, self.name)

    def tearDown(self) -> None:
        """
        Restore back write rights on created paths: if tests modified the
        rights, that will allow the paths to be removed easily afterwards.
        """
        os.chmod(self.dir, 511)
        if os.path.exists(self.path):
            os.chmod(self.path, 511)

    def test_abstractShouldRotate(self) -> None:
        """
        L{BaseLogFile.shouldRotate} is abstract and must be implemented by
        subclass.
        """
        log = logfile.BaseLogFile(self.name, self.dir)
        self.addCleanup(log.close)
        self.assertRaises(NotImplementedError, log.shouldRotate)

    def test_writing(self) -> None:
        """
        Log files can be written to, flushed and closed. Closing a log file
        also flushes it.
        """
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log:
            log.write('123')
            log.write('456')
            log.flush()
            log.write('7890')
        with open(self.path) as f:
            self.assertEqual(f.read(), '1234567890')

    def test_rotation(self) -> None:
        """
        Rotating log files autorotate after a period of time, and can also be
        manually rotated.
        """
        with contextlib.closing(logfile.LogFile(self.name, self.dir, rotateLength=10)) as log:
            log.write('123')
            log.write('4567890')
            log.write('1' * 11)
            self.assertTrue(os.path.exists(f'{self.path}.1'))
            self.assertFalse(os.path.exists(f'{self.path}.2'))
            log.write('')
            self.assertTrue(os.path.exists(f'{self.path}.1'))
            self.assertTrue(os.path.exists(f'{self.path}.2'))
            self.assertFalse(os.path.exists(f'{self.path}.3'))
            log.write('3')
            self.assertFalse(os.path.exists(f'{self.path}.3'))
            log.rotate()
            self.assertTrue(os.path.exists(f'{self.path}.3'))
            self.assertFalse(os.path.exists(f'{self.path}.4'))
        self.assertEqual(log.listLogs(), [1, 2, 3])

    def test_append(self) -> None:
        """
        Log files can be written to, closed. Their size is the number of
        bytes written to them. Everything that was written to them can
        be read, even if the writing happened on separate occasions,
        and even if the log file was closed in between.
        """
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log:
            log.write('0123456789')
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        self.assertEqual(log.size, 10)
        self.assertEqual(log._file.tell(), log.size)
        log.write('abc')
        self.assertEqual(log.size, 13)
        self.assertEqual(log._file.tell(), log.size)
        f = log._file
        f.seek(0, 0)
        self.assertEqual(f.read(), b'0123456789abc')

    def test_logReader(self) -> None:
        """
        Various tests for log readers.

        First of all, log readers can get logs by number and read what
        was written to those log files. Getting nonexistent log files
        raises C{ValueError}. Using anything other than an integer
        index raises C{TypeError}. As logs get older, their log
        numbers increase.
        """
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc\n')
        log.write('def\n')
        log.rotate()
        log.write('ghi\n')
        log.flush()
        self.assertEqual(log.listLogs(), [1])
        with contextlib.closing(log.getCurrentLog()) as reader:
            reader._file.seek(0)
            self.assertEqual(reader.readLines(), ['ghi\n'])
            self.assertEqual(reader.readLines(), [])
        with contextlib.closing(log.getLog(1)) as reader:
            self.assertEqual(reader.readLines(), ['abc\n', 'def\n'])
            self.assertEqual(reader.readLines(), [])
        self.assertRaises(ValueError, log.getLog, 2)
        self.assertRaises(TypeError, log.getLog, '1')
        log.rotate()
        self.assertEqual(log.listLogs(), [1, 2])
        with contextlib.closing(log.getLog(1)) as reader:
            reader._file.seek(0)
            self.assertEqual(reader.readLines(), ['ghi\n'])
            self.assertEqual(reader.readLines(), [])
        with contextlib.closing(log.getLog(2)) as reader:
            self.assertEqual(reader.readLines(), ['abc\n', 'def\n'])
            self.assertEqual(reader.readLines(), [])

    def test_LogReaderReadsZeroLine(self) -> None:
        """
        L{LogReader.readLines} supports reading no line.
        """
        with open(self.path, 'w'):
            pass
        reader = logfile.LogReader(self.path)
        self.addCleanup(reader.close)
        self.assertEqual([], reader.readLines(0))

    def test_modePreservation(self) -> None:
        """
        Check rotated files have same permissions as original.
        """
        open(self.path, 'w').close()
        os.chmod(self.path, 455)
        mode = os.stat(self.path)[stat.ST_MODE]
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc')
        log.rotate()
        self.assertEqual(mode, os.stat(self.path)[stat.ST_MODE])

    def test_noPermission(self) -> None:
        """
        Check it keeps working when permission on dir changes.
        """
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        log.write('abc')
        os.chmod(self.dir, 365)
        try:
            f = open(os.path.join(self.dir, 'xxx'), 'w')
        except OSError:
            pass
        else:
            f.close()
            return
        log.rotate()
        log.write('def')
        log.flush()
        f = log._file
        self.assertEqual(f.tell(), 6)
        f.seek(0, 0)
        self.assertEqual(f.read(), b'abcdef')

    def test_maxNumberOfLog(self) -> None:
        """
        Test it respect the limit on the number of files when maxRotatedFiles
        is not None.
        """
        log = logfile.LogFile(self.name, self.dir, rotateLength=10, maxRotatedFiles=3)
        self.addCleanup(log.close)
        log.write('1' * 11)
        log.write('2' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.1'))
        log.write('3' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.2'))
        log.write('4' * 11)
        self.assertTrue(os.path.exists(f'{self.path}.3'))
        with open(f'{self.path}.3') as fp:
            self.assertEqual(fp.read(), '1' * 11)
        log.write('5' * 11)
        with open(f'{self.path}.3') as fp:
            self.assertEqual(fp.read(), '2' * 11)
        self.assertFalse(os.path.exists(f'{self.path}.4'))

    def test_fromFullPath(self) -> None:
        """
        Test the fromFullPath method.
        """
        log1 = logfile.LogFile(self.name, self.dir, 10, defaultMode=511)
        self.addCleanup(log1.close)
        log2 = logfile.LogFile.fromFullPath(self.path, 10, defaultMode=511)
        self.addCleanup(log2.close)
        self.assertEqual(log1.name, log2.name)
        self.assertEqual(os.path.abspath(log1.path), log2.path)
        self.assertEqual(log1.rotateLength, log2.rotateLength)
        self.assertEqual(log1.defaultMode, log2.defaultMode)

    def test_defaultPermissions(self) -> None:
        """
        Test the default permission of the log file: if the file exist, it
        should keep the permission.
        """
        with open(self.path, 'wb'):
            os.chmod(self.path, 455)
            currentMode = stat.S_IMODE(os.stat(self.path)[stat.ST_MODE])
        log1 = logfile.LogFile(self.name, self.dir)
        self.assertEqual(stat.S_IMODE(os.stat(self.path)[stat.ST_MODE]), currentMode)
        self.addCleanup(log1.close)

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

    @skipIf(runtime.platform.isWindows(), "Can't test reopen on Windows")
    def test_reopen(self) -> None:
        """
        L{logfile.LogFile.reopen} allows to rename the currently used file and
        make L{logfile.LogFile} create a new file.
        """
        with contextlib.closing(logfile.LogFile(self.name, self.dir)) as log1:
            log1.write('hello1')
            savePath = os.path.join(self.dir, 'save.log')
            os.rename(self.path, savePath)
            log1.reopen()
            log1.write('hello2')
        with open(self.path) as f:
            self.assertEqual(f.read(), 'hello2')
        with open(savePath) as f:
            self.assertEqual(f.read(), 'hello1')

    def test_nonExistentDir(self) -> None:
        """
        Specifying an invalid directory to L{LogFile} raises C{IOError}.
        """
        e = self.assertRaises(IOError, logfile.LogFile, self.name, 'this_dir_does_not_exist')
        self.assertEqual(e.errno, errno.ENOENT)

    def test_cantChangeFileMode(self) -> None:
        """
        Opening a L{LogFile} which can be read and write but whose mode can't
        be changed doesn't trigger an error.
        """
        if runtime.platform.isWindows():
            name, directory = ('NUL', '')
            expectedPath = 'NUL'
        else:
            name, directory = ('null', '/dev')
            expectedPath = '/dev/null'
        log = logfile.LogFile(name, directory, defaultMode=365)
        self.addCleanup(log.close)
        self.assertEqual(log.path, expectedPath)
        self.assertEqual(log.defaultMode, 365)

    def test_listLogsWithBadlyNamedFiles(self) -> None:
        """
        L{LogFile.listLogs} doesn't choke if it encounters a file with an
        unexpected name.
        """
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        with open(f'{log.path}.1', 'w') as fp:
            fp.write('123')
        with open(f'{log.path}.bad-file', 'w') as fp:
            fp.write('123')
        self.assertEqual([1], log.listLogs())

    def test_listLogsIgnoresZeroSuffixedFiles(self) -> None:
        """
        L{LogFile.listLogs} ignores log files which rotated suffix is 0.
        """
        log = logfile.LogFile(self.name, self.dir)
        self.addCleanup(log.close)
        for i in range(0, 3):
            with open(f'{log.path}.{i}', 'w') as fp:
                fp.write('123')
        self.assertEqual([1, 2], log.listLogs())