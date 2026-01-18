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
def test_saveLoad(self) -> None:
    """
        Saving and loading a dictionary with some simple values in it results
        in those same simple values in the output; according to JSON's rules,
        though, all dictionary keys must be L{str} and any non-L{str}
        keys will be converted.
        """
    self.assertEqual(eventFromJSON(self.savedEventJSON({1: 2, '3': '4'})), {'1': 2, '3': '4'})