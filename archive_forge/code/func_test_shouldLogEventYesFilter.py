from typing import Iterable, List, Tuple, Union, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from constantly import NamedConstant
from twisted.trial import unittest
from .._filter import (
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._observer import LogPublisher, bitbucketLogObserver
def test_shouldLogEventYesFilter(self) -> None:
    """
        Filter with positive predicate result.
        """
    self.assertEqual(self.filterWith(['twoPlus']), [0, 1, 2, 3])