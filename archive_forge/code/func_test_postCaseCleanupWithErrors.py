from __future__ import annotations
import locale
import os
import sys
from io import StringIO
from typing import Generator
from zope.interface import implementer
from hamcrest import assert_that, equal_to
from twisted.internet.base import DelayedCall
from twisted.internet.interfaces import IProcessTransport
from twisted.python import filepath
from twisted.python.failure import Failure
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase
from twisted.trial.util import (
def test_postCaseCleanupWithErrors(self) -> None:
    """
        The post-case cleanup method will return False and call C{addError} on
        the result with a L{DirtyReactorAggregateError} Failure if there are
        pending calls.
        """
    delayedCall = DelayedCall(300, lambda: None, (), {}, lambda x: None, lambda x: None, seconds=lambda: 0)
    delayedCallString = str(delayedCall)
    reactor = StubReactor([delayedCall], [])
    test = object()
    reporter = StubErrorReporter()
    jan = _Janitor(test, reporter, reactor=reactor)
    self.assertFalse(jan.postCaseCleanup())
    self.assertEqual(len(reporter.errors), 1)
    self.assertEqual(reporter.errors[0][1].value.delayedCalls, [delayedCallString])