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
def test_readEventsAutoWithRecordSeparator(self) -> None:
    """
        L{eventsFromJSONLogFile} reads events from a file and automatically
        detects use of C{"\\x1e"} as the record separator.
        """
    with StringIO('\x1e{"x": 1}\n\x1e{"y": 2}\n') as fileHandle:
        self._readEvents(fileHandle)
        self.assertEqual(len(self.errorEvents), 0)