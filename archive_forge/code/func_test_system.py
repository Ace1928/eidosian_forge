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
def test_system(self) -> None:
    """
        The old-style C{"system"} key is copied to the new-style
        C{"log_system"} key.
        """
    publishToNewObserver(self.observer, self.legacyEvent(), legacyLog.textFromEventDict)
    self.assertEqual(self.events[0]['log_system'], self.events[0]['system'])