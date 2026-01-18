from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def test_returnsFileLogObserver(self) -> None:
    """
        L{textFileLogObserver} returns a L{FileLogObserver}.
        """
    with StringIO() as fileHandle:
        observer = textFileLogObserver(fileHandle)
        self.assertIsInstance(observer, FileLogObserver)