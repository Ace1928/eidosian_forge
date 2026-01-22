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
class TodoTests(unittest.SynchronousTestCase):
    """
    Tests for L{reporter.Reporter}'s handling of todos.
    """

    def setUp(self):
        self.stream = StringIO()
        self.result = reporter.Reporter(self.stream)
        self.test = sample.FooTest('test_foo')

    def _getTodos(self, result):
        """
        Get the expected failures that happened to a reporter.
        """
        return result.expectedFailures

    def _getUnexpectedSuccesses(self, result):
        """
        Get the unexpected successes that happened to a reporter.
        """
        return result.unexpectedSuccesses

    def test_accumulation(self):
        """
        L{reporter.Reporter} accumulates the expected failures that it
        is notified of.
        """
        self.result.addExpectedFailure(self.test, Failure(Exception()), makeTodo('todo!'))
        self.assertEqual(len(self._getTodos(self.result)), 1)

    def test_noTodoProvided(self):
        """
        If no C{Todo} is provided to C{addExpectedFailure}, then
        L{reporter.Reporter} makes up a sensible default.

        This allows standard Python unittests to use Twisted reporters.
        """
        failure = Failure(Exception())
        self.result.addExpectedFailure(self.test, failure)
        [(test, error, todo)] = self._getTodos(self.result)
        self.assertEqual(test, self.test)
        self.assertEqual(error, failure)
        self.assertEqual(repr(todo), repr(makeTodo('Test expected to fail')))

    def test_success(self):
        """
        A test run is still successful even if there are expected failures.
        """
        self.result.addExpectedFailure(self.test, Failure(Exception()), makeTodo('todo!'))
        self.assertEqual(True, self.result.wasSuccessful())

    def test_unexpectedSuccess(self):
        """
        A test which is marked as todo but succeeds will have an unexpected
        success reported to its result. A test run is still successful even
        when this happens.
        """
        self.result.addUnexpectedSuccess(self.test, makeTodo('Heya!'))
        self.assertEqual(True, self.result.wasSuccessful())
        self.assertEqual(len(self._getUnexpectedSuccesses(self.result)), 1)

    def test_unexpectedSuccessNoTodo(self):
        """
        A test which is marked as todo but succeeds will have an unexpected
        success reported to its result. A test run is still successful even
        when this happens.

        If no C{Todo} is provided, then we make up a sensible default. This
        allows standard Python unittests to use Twisted reporters.
        """
        self.result.addUnexpectedSuccess(self.test)
        [(test, todo)] = self._getUnexpectedSuccesses(self.result)
        self.assertEqual(test, self.test)
        self.assertEqual(repr(todo), repr(makeTodo('Test expected to fail')))

    def test_summary(self):
        """
        The reporter's C{printSummary} method should print the number of
        expected failures that occurred.
        """
        self.result.addExpectedFailure(self.test, Failure(Exception()), makeTodo('some reason'))
        self.result.done()
        output = self.stream.getvalue().splitlines()[-1]
        prefix = 'PASSED '
        self.assertTrue(output.startswith(prefix))
        self.assertEqual(output[len(prefix):].strip(), '(expectedFailures=1)')

    def test_basicErrors(self):
        """
        The reporter's L{printErrors} method should include the value of the
        Todo.
        """
        self.result.addExpectedFailure(self.test, Failure(Exception()), makeTodo('some reason'))
        self.result.done()
        output = self.stream.getvalue().splitlines()[3].strip()
        self.assertEqual(output, "Reason: 'some reason'")

    def test_booleanTodo(self):
        """
        Booleans CAN'T be used as the value of a todo. Maybe this sucks. This
        is a test for current behavior, not a requirement.
        """
        self.result.addExpectedFailure(self.test, Failure(Exception()), True)
        self.assertRaises(Exception, self.result.done)

    def test_exceptionTodo(self):
        """
        The exception for expected failures should be shown in the
        C{printErrors} output.
        """
        try:
            1 / 0
        except Exception as e:
            error = e
        self.result.addExpectedFailure(self.test, Failure(error), makeTodo('todo!'))
        self.result.done()
        output = '\n'.join(self.stream.getvalue().splitlines()[3:]).strip()
        self.assertTrue(str(error) in output)

    def test_standardLibraryCompatibilityFailure(self):
        """
        Tests that use the standard library C{expectedFailure} feature worth
        with Trial reporters.
        """

        class Test(StdlibTestCase):

            @expectedFailure
            def test_fail(self):
                self.fail('failure')
        test = Test('test_fail')
        test.run(self.result)
        self.assertEqual(len(self._getTodos(self.result)), 1)

    def test_standardLibraryCompatibilitySuccess(self):
        """
        Tests that use the standard library C{expectedFailure} feature worth
        with Trial reporters.
        """

        class Test(StdlibTestCase):

            @expectedFailure
            def test_success(self):
                pass
        test = Test('test_success')
        test.run(self.result)
        self.assertEqual(len(self._getUnexpectedSuccesses(self.result)), 1)