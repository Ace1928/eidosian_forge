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
def test_saveLoadUnknownLevel(self) -> None:
    """
        If a saved bit of JSON (let's say, from a future version of Twisted)
        were to persist a different log_level, it will resolve as None.
        """
    loadedEvent = eventFromJSON('{"log_level": {"name": "other", "__class_uuid__": "02E59486-F24D-46AD-8224-3ACDF2A5732A"}}')
    self.assertEqual(loadedEvent, dict(log_level=None))