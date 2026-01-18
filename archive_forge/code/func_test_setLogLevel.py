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
def test_setLogLevel(self) -> None:
    """
        Setting and retrieving log levels.
        """
    predicate = LogLevelFilterPredicate()
    for default in ('', cast(str, None)):
        predicate.setLogLevelForNamespace(default, LogLevel.error)
        predicate.setLogLevelForNamespace('twext.web2', LogLevel.debug)
        predicate.setLogLevelForNamespace('twext.web2.dav', LogLevel.warn)
        self.assertEqual(predicate.logLevelForNamespace(''), LogLevel.error)
        self.assertEqual(predicate.logLevelForNamespace(cast(str, None)), LogLevel.error)
        self.assertEqual(predicate.logLevelForNamespace('twisted'), LogLevel.error)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2'), LogLevel.debug)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav'), LogLevel.warn)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test'), LogLevel.warn)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test1.test2'), LogLevel.warn)