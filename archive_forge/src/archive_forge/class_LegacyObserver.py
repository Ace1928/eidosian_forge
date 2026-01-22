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
@implementer(legacyLog.ILogObserver)
class LegacyObserver:

    def __repr__(self) -> str:
        return '<Legacy Observer>'

    def __call__(self, eventDict: legacyLog.EventDict) -> None:
        return