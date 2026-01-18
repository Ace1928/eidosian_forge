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
def test_readInvalidUTF8Bytes(self) -> None:
    """
        If the JSON text for a record contains invalid UTF-8 text, ignore that
        record.
        """
    with BytesIO(b'\x1e{"x": "\xe2\xac"}\n\x1e{"y": 2}\n') as fileHandle:
        events = iter(eventsFromJSONLogFile(fileHandle))
        self.assertEqual(next(events), {'y': 2})
        self.assertRaises(StopIteration, next, events)
        self.assertEqual(len(self.errorEvents), 1)
        self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to decode UTF-8 for JSON record: {record!r}')
        self.assertEqual(self.errorEvents[0]['record'], b'{"x": "\xe2\xac"}\n')