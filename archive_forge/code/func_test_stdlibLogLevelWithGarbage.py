import logging as py_logging
from time import time
from typing import List, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python import context, log as legacyLog
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._format import formatEvent
from .._interfaces import ILogObserver, LogEvent
from .._legacy import LegacyLogObserverWrapper, publishToNewObserver
from .._levels import LogLevel
def test_stdlibLogLevelWithGarbage(self) -> None:
    """
        If the old-style C{"logLevel"} key is set to a standard library logging
        level, using an unknown value, the new-style C{"log_level"} key should
        not get set.
        """
    publishToNewObserver(self.observer, self.legacyEvent(logLevel='Foo!!!!!'), legacyLog.textFromEventDict)
    self.assertNotIn('log_level', self.events[0])