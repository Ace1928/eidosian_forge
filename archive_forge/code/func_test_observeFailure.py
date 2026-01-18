from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
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