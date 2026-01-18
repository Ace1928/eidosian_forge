from typing import List, cast
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.trial import unittest
from .._buffer import LimitedHistoryLogObserver
from .._interfaces import ILogObserver, LogEvent

        When more events than a L{LimitedHistoryLogObserver}'s maximum size are
        buffered, older events will be dropped.
        