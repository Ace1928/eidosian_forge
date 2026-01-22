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
class ErrorReportingTests(StringTest):
    doubleSeparator = re.compile('^=+$')

    def setUp(self) -> None:
        self.loader = runner.TestLoader()
        self.output = StringIO()
        self.result: reporter.Reporter = reporter.Reporter(self.output)

    def getOutput(self, suite):
        result = self.getResult(suite)
        result.done()
        return self.output.getvalue()

    def getResult(self, suite: PyUnitTestSuite) -> reporter.Reporter:
        suite.run(self.result)
        return self.result

    def test_formatErroredMethod(self):
        """
        A test method which runs and has an error recorded against it is
        reported in the output stream with the I{ERROR} tag along with a
        summary of what error was reported and the ID of the test.
        """
        cls = erroneous.SynchronousTestFailureInSetUp
        suite = self.loader.loadClass(cls)
        output = self.getOutput(suite).splitlines()
        match = [self.doubleSeparator, '[ERROR]', 'Traceback (most recent call last):', re.compile('^\\s+File .*erroneous\\.py., line \\d+, in setUp$'), re.compile('^\\s+raise FoolishError..I am a broken setUp method..$'), 'twisted.trial.test.erroneous.FoolishError: I am a broken setUp method', f'{cls.__module__}.{cls.__name__}.test_noop']
        self.stringComparison(match, output)

    def test_formatFailedMethod(self):
        """
        A test method which runs and has a failure recorded against it is
        reported in the output stream with the I{FAIL} tag along with a summary
        of what failure was reported and the ID of the test.
        """
        suite = self.loader.loadByName('twisted.trial.test.erroneous.TestRegularFail.test_fail')
        output = self.getOutput(suite).splitlines()
        match = [self.doubleSeparator, '[FAIL]', 'Traceback (most recent call last):', re.compile('^\\s+File .*erroneous\\.py., line \\d+, in test_fail$'), re.compile('^\\s+self\\.fail\\("I fail"\\)$'), 'twisted.trial.unittest.FailTest: I fail', 'twisted.trial.test.erroneous.TestRegularFail.test_fail']
        self.stringComparison(match, output)

    def test_doctestError(self):
        """
        A problem encountered while running a doctest is reported in the output
        stream with a I{FAIL} or I{ERROR} tag along with a summary of what
        problem was encountered and the ID of the test.
        """
        from twisted.trial.test import erroneous
        suite = unittest.decorate(self.loader.loadDoctests(erroneous), itrial.ITestCase)
        output = self.getOutput(suite)
        path = 'twisted.trial.test.erroneous.unexpectedException'
        for substring in ['1/0', 'ZeroDivisionError', 'Exception raised:', path]:
            self.assertSubstring(substring, output)
        self.assertTrue(re.search('Fail(ed|ure in) example:', output), "Couldn't match 'Failure in example: ' or 'Failed example: '")
        expect = [self.doubleSeparator, re.compile('\\[(ERROR|FAIL)\\]')]
        self.stringComparison(expect, output.splitlines())

    def test_hiddenException(self) -> None:
        """
        When a function scheduled using L{IReactorTime.callLater} in a
        test method raises an exception that exception is added to the test
        result as an error.

        This happens even if the test also fails and the test failure is also
        added to the test result as a failure.

        Only really necessary for testing the deprecated style of tests that
        use iterate() directly. See
        L{erroneous.DelayedCall.testHiddenException} for more details.
        """
        test = erroneous.DelayedCall('testHiddenException')
        result = self.getResult(PyUnitTestSuite([test]))
        assert_that(result, matches_result(errors=has_length(1), failures=has_length(1)))
        [(actualCase, error)] = result.errors
        assert_that(test, equal_to(actualCase))
        assert_that(error, isFailure(type=equal_to(RuntimeError), value=after(str, equal_to('something blew up')), frames=has_item(similarFrame('go', 'erroneous.py'))))
        [(actualCase, failure)] = result.failures
        assert_that(test, equal_to(actualCase))
        assert_that(failure, isFailure(type=equal_to(test.failureException), value=after(str, equal_to('Deliberate failure to mask the hidden exception')), frames=has_item(similarFrame('testHiddenException', 'erroneous.py'))))