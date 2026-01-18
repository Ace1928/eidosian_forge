from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def test_observeWritesEmpty(self) -> None:
    """
        L{FileLogObserver} does not write to the given file when it observes
        events and C{formatEvent} returns C{""}.
        """
    self._test_observeWrites('', 0)