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
def test_pythonLogLevelNotSet(self) -> None:
    """
        The new-style C{"log_level"} key is not translated to the old-style
        C{"logLevel"} key.

        Events are forwarded from the old module from to new module and are
        then seen by old-style observers.
        We don't want to add unexpected keys to old-style events.
        """
    event = self.forwardAndVerify(dict(log_level=LogLevel.info))
    self.assertNotIn('logLevel', event)