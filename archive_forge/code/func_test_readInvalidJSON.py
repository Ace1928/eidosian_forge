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
def test_readInvalidJSON(self) -> None:
    """
        If the JSON text for a record is invalid, skip it.
        """
    with StringIO('\x1e{"x": }\n\x1e{"y": 2}\n') as fileHandle:
        events = iter(eventsFromJSONLogFile(fileHandle))
        self.assertEqual(next(events), {'y': 2})
        self.assertRaises(StopIteration, next, events)
        self.assertEqual(len(self.errorEvents), 1)
        self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to read JSON record: {record!r}')
        self.assertEqual(self.errorEvents[0]['record'], b'{"x": }\n')