import os
from io import BytesIO, StringIO
from typing import Type
from unittest import TestCase as PyUnitTestCase
from zope.interface.verify import verifyObject
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.internet.defer import Deferred, fail
from twisted.internet.error import ConnectionLost, ProcessDone
from twisted.internet.interfaces import IAddress, ITransport
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.trial._dist import managercommands
from twisted.trial._dist.worker import (
from twisted.trial.reporter import TestResult
from twisted.trial.test import pyunitcases, skipping
from twisted.trial.unittest import TestCase, makeTodo
from .matchers import isFailure, matches_result, similarFrame
def test_runUnexpectedSuccesses(self) -> None:
    """
        Run a test, and succeed unexpectedly.
        """
    expectedCase = skipping.SynchronousStrictTodo('test_todo7')
    result = self.workerRunTest(expectedCase)
    assert_that(result, matches_result(unexpectedSuccesses=has_length(1)))
    [(actualCase, unexpectedSuccess)] = result.unexpectedSuccesses
    assert_that(expectedCase, equal_to(actualCase))
    assert_that(unexpectedSuccess, equal_to('todo7'))