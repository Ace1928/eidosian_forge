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
def test_saveBytes(self) -> None:
    """
        Any L{bytes} objects will be saved as if they are latin-1 so they can
        be faithfully re-loaded.
        """
    inputEvent = {'hello': bytes(range(255))}
    inputEvent.update({b'skipped': 'okay'})
    self.assertEqual(eventFromJSON(self.savedEventJSON(inputEvent)), {'hello': bytes(range(255)).decode('charmap')})