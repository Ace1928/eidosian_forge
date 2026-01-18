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
def test_shouldLogEventYesYesNoFilter(self) -> None:
    """
        Series of filters with positive, positive and negative predicate
        results.
        """
    self.assertEqual(self.filterWith(['twoPlus', 'twoMinus', 'no']), [0, 1, 2, 3])