from io import StringIO
from types import TracebackType
from typing import IO, Any, AnyStr, Optional, Type, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._file import FileLogObserver, textFileLogObserver
from .._interfaces import ILogObserver
def test_observeFlushes(self) -> None:
    """
        L{FileLogObserver} calles C{flush()} on the output file when it
        observes an event.
        """
    with DummyFile() as fileHandle:
        observer = FileLogObserver(cast(IO[Any], fileHandle), lambda e: str(e))
        event = dict(x=1)
        observer(event)
        self.assertEqual(fileHandle.flushes, 1)