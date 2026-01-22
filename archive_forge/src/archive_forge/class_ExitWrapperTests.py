import errno
import os
import re
import sys
from inspect import getmro
from io import BytesIO, StringIO
from typing import Type
from unittest import (
from hamcrest import assert_that, equal_to, has_item, has_length
from twisted.python import log
from twisted.python.failure import Failure
from twisted.trial import itrial, reporter, runner, unittest, util
from twisted.trial.reporter import UncleanWarningsReporterWrapper, _ExitWrapper
from twisted.trial.test import erroneous, sample
from twisted.trial.unittest import SkipTest, Todo, makeTodo
from .._dist.test.matchers import isFailure, matches_result, similarFrame
from .matchers import after
class ExitWrapperTests(unittest.SynchronousTestCase):
    """
    Tests for L{reporter._ExitWrapper}.
    """

    def setUp(self):
        self.failure = Failure(Exception('I am a Failure'))
        self.test = sample.FooTest('test_foo')
        self.result = reporter.TestResult()
        self.wrapped = _ExitWrapper(self.result)
        self.assertFalse(self.wrapped.shouldStop)

    def test_stopOnFailure(self):
        """
        L{reporter._ExitWrapper} causes a wrapped reporter to stop after its
        first failure.
        """
        self.wrapped.addFailure(self.test, self.failure)
        self.assertTrue(self.wrapped.shouldStop)
        self.assertEqual(self.result.failures, [(self.test, self.failure)])

    def test_stopOnError(self):
        """
        L{reporter._ExitWrapper} causes a wrapped reporter to stop after its
        first error.
        """
        self.wrapped.addError(self.test, self.failure)
        self.assertTrue(self.wrapped.shouldStop)
        self.assertEqual(self.result.errors, [(self.test, self.failure)])

    def test_doesNotStopOnUnexpectedSuccess(self):
        """
        L{reporter._ExitWrapper} does not cause a wrapped reporter to stop
        after an unexpected success.
        """
        self.wrapped.addUnexpectedSuccess(self.test, self.failure)
        self.assertFalse(self.wrapped.shouldStop)
        self.assertEqual(self.result.unexpectedSuccesses, [(self.test, self.failure)])