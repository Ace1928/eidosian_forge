from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
class TextFileLogObserverTests(TestCase):
    """
    Tests for L{textFileLogObserver}.
    """

    def test_returnsFileLogObserver(self) -> None:
        """
        L{textFileLogObserver} returns a L{FileLogObserver}.
        """
        with StringIO() as fileHandle:
            observer = textFileLogObserver(fileHandle)
            self.assertIsInstance(observer, FileLogObserver)

    def test_outFile(self) -> None:
        """
        Returned L{FileLogObserver} has the correct outFile.
        """
        with StringIO() as fileHandle:
            observer = textFileLogObserver(fileHandle)
            self.assertIs(observer._outFile, fileHandle)

    def test_timeFormat(self) -> None:
        """
        Returned L{FileLogObserver} has the correct outFile.
        """
        with StringIO() as fileHandle:
            observer = textFileLogObserver(fileHandle, timeFormat='%f')
            observer(dict(log_format='XYZZY', log_time=112345.6))
            self.assertEqual(fileHandle.getvalue(), '600000 [-#-] XYZZY\n')

    def test_observeFailure(self) -> None:
        """
        If the C{"log_failure"} key exists in an event, the observer appends
        the failure's traceback to the output.
        """
        with StringIO() as fileHandle:
            observer = textFileLogObserver(fileHandle)
            try:
                1 / 0
            except ZeroDivisionError:
                failure = Failure()
            event = dict(log_failure=failure)
            observer(event)
            output = fileHandle.getvalue()
            self.assertTrue(output.split('\n')[1].startswith('\tTraceback '), msg=repr(output))

    def test_observeFailureThatRaisesInGetTraceback(self) -> None:
        """
        If the C{"log_failure"} key exists in an event, and contains an object
        that raises when you call its C{getTraceback()}, then the observer
        appends a message noting the problem, instead of raising.
        """
        with StringIO() as fileHandle:
            observer = textFileLogObserver(fileHandle)
            event = dict(log_failure=object())
            observer(event)
            output = fileHandle.getvalue()
            expected = '(UNABLE TO OBTAIN TRACEBACK FROM EVENT)'
            self.assertIn(expected, output)