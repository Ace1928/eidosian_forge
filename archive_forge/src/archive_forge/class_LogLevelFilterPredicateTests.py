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
class LogLevelFilterPredicateTests(unittest.TestCase):
    """
    Tests for L{LogLevelFilterPredicate}.
    """

    def test_defaultLogLevel(self) -> None:
        """
        Default log level is used.
        """
        predicate = LogLevelFilterPredicate()
        for default in ('', cast(str, None)):
            self.assertEqual(predicate.logLevelForNamespace(default), predicate.defaultLogLevel)
            self.assertEqual(predicate.logLevelForNamespace('rocker.cool.namespace'), predicate.defaultLogLevel)

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

    def test_setInvalidLogLevel(self) -> None:
        """
        Can't pass invalid log levels to C{setLogLevelForNamespace()}.
        """
        predicate = LogLevelFilterPredicate()
        self.assertRaises(InvalidLogLevelError, predicate.setLogLevelForNamespace, 'twext.web2', object())
        self.assertRaises(InvalidLogLevelError, predicate.setLogLevelForNamespace, 'twext.web2', 'debug')

    def test_clearLogLevels(self) -> None:
        """
        Clearing log levels.
        """
        predicate = LogLevelFilterPredicate()
        predicate.setLogLevelForNamespace('twext.web2', LogLevel.debug)
        predicate.setLogLevelForNamespace('twext.web2.dav', LogLevel.error)
        predicate.clearLogLevels()
        self.assertEqual(predicate.logLevelForNamespace('twisted'), predicate.defaultLogLevel)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2'), predicate.defaultLogLevel)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav'), predicate.defaultLogLevel)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test'), predicate.defaultLogLevel)
        self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test1.test2'), predicate.defaultLogLevel)

    def test_filtering(self) -> None:
        """
        Events are filtered based on log level/namespace.
        """
        predicate = LogLevelFilterPredicate()
        predicate.setLogLevelForNamespace('', LogLevel.error)
        predicate.setLogLevelForNamespace('twext.web2', LogLevel.debug)
        predicate.setLogLevelForNamespace('twext.web2.dav', LogLevel.warn)

        def checkPredicate(namespace: str, level: NamedConstant, expectedResult: NamedConstant) -> None:
            event: LogEvent = dict(log_namespace=namespace, log_level=level)
            self.assertEqual(expectedResult, predicate(event))
        checkPredicate('', LogLevel.debug, PredicateResult.no)
        checkPredicate(cast(str, None), LogLevel.debug, PredicateResult.no)
        checkPredicate('', LogLevel.error, PredicateResult.no)
        checkPredicate(cast(str, None), LogLevel.error, PredicateResult.no)
        checkPredicate('twext.web2', LogLevel.debug, PredicateResult.maybe)
        checkPredicate('twext.web2', LogLevel.error, PredicateResult.maybe)
        checkPredicate('twext.web2.dav', LogLevel.debug, PredicateResult.no)
        checkPredicate('twext.web2.dav', LogLevel.error, PredicateResult.maybe)
        checkPredicate('', LogLevel.critical, PredicateResult.no)
        checkPredicate(cast(str, None), LogLevel.critical, PredicateResult.no)
        checkPredicate('twext.web2', None, PredicateResult.no)