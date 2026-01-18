from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_sourceAvailableForFormatting(self) -> None:
    """
        On instances that have a L{Logger} class attribute, the C{log_source}
        key is available to format strings.
        """
    obj = LogComposedObject('hello')
    log = cast(TestLogger, obj.log)
    log.error('Hello, {log_source}.')
    self.assertIn('log_source', log.event)
    self.assertEqual(log.event['log_source'], obj)
    stuff = formatEvent(log.event)
    self.assertIn('Hello, <LogComposedObject hello>.', stuff)