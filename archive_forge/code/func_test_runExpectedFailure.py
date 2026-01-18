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
def test_runExpectedFailure(self) -> None:
    """
        Run a test, and fail expectedly.
        """
    expectedCase = skipping.SynchronousStrictTodo('test_todo1')
    result = self.workerRunTest(expectedCase)
    assert_that(result, matches_result(expectedFailures=has_length(1)))
    [(actualCase, exceptionMessage, todoReason)] = result.expectedFailures
    assert_that(actualCase, equal_to(expectedCase))
    assert_that(exceptionMessage, equal_to('expected failure'))
    assert_that(todoReason, equal_to(makeTodo('todo1')))