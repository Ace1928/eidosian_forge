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
def test_readUnseparated(self) -> None:
    """
        Multiple events without a record separator are skipped.
        """
    with StringIO('\x1e{"x": 1}\n{"y": 2}\n') as fileHandle:
        events = eventsFromJSONLogFile(fileHandle)
        self.assertRaises(StopIteration, next, events)
        self.assertEqual(len(self.errorEvents), 1)
        self.assertEqual(self.errorEvents[0]['log_format'], 'Unable to read JSON record: {record!r}')
        self.assertEqual(self.errorEvents[0]['record'], b'{"x": 1}\n{"y": 2}\n')