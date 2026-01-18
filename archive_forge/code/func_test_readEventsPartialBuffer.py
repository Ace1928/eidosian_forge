from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
def test_readEventsPartialBuffer(self) -> None:
    """
        L{eventsFromJSONLogFile} handles buffering a partial event.
        """
    with StringIO('\x1e{"x": 1}\n\x1e{"y": 2}\n') as fileHandle:
        self._readEvents(fileHandle, bufferSize=1)
        self.assertEqual(len(self.errorEvents), 0)