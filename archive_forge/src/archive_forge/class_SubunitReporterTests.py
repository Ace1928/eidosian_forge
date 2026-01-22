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
class SubunitReporterTests(ReporterInterfaceTests):
    """
    Tests for the subunit reporter.

    This just tests that the subunit reporter implements the basic interface.
    """
    resultFactory: Type[itrial.IReporter] = reporter.SubunitReporter

    def setUp(self):
        if reporter.TestProtocolClient is None:
            raise SkipTest('Subunit not installed, cannot test SubunitReporter')
        self.test = sample.FooTest('test_foo')
        self.stream = BytesIO()
        self.publisher = log.LogPublisher()
        self.result = self.resultFactory(self.stream, publisher=self.publisher)

    def assertForwardsToSubunit(self, methodName, *args, **kwargs):
        """
        Assert that 'methodName' on L{SubunitReporter} forwards to the
        equivalent method on subunit.

        Checks that the return value from subunit is returned from the
        L{SubunitReporter} and that the reporter writes the same data to its
        stream as subunit does to its own.

        Assumes that the method on subunit has the same name as the method on
        L{SubunitReporter}.
        """
        stream = BytesIO()
        subunitClient = reporter.TestProtocolClient(stream)
        subunitReturn = getattr(subunitClient, methodName)(*args, **kwargs)
        subunitOutput = stream.getvalue()
        reporterReturn = getattr(self.result, methodName)(*args, **kwargs)
        self.assertEqual(subunitReturn, reporterReturn)
        self.assertEqual(subunitOutput, self.stream.getvalue())

    def removeMethod(self, klass, methodName):
        """
        Remove 'methodName' from 'klass'.

        If 'klass' does not have a method named 'methodName', then
        'removeMethod' succeeds silently.

        If 'klass' does have a method named 'methodName', then it is removed
        using delattr. Also, methods of the same name are removed from all
        base classes of 'klass', thus removing the method entirely.

        @param klass: The class to remove the method from.
        @param methodName: The name of the method to remove.
        """
        method = getattr(klass, methodName, None)
        if method is None:
            return
        for base in getmro(klass):
            try:
                delattr(base, methodName)
            except (AttributeError, TypeError):
                break
            else:
                self.addCleanup(setattr, base, methodName, method)

    def test_subunitWithoutAddExpectedFailureInstalled(self):
        """
        Some versions of subunit don't have "addExpectedFailure". For these
        versions, we report expected failures as successes.
        """
        self.removeMethod(reporter.TestProtocolClient, 'addExpectedFailure')
        try:
            1 / 0
        except ZeroDivisionError:
            self.result.addExpectedFailure(self.test, sys.exc_info(), 'todo')
        expectedFailureOutput = self.stream.getvalue()
        self.stream.truncate(0)
        self.stream.seek(0)
        self.result.addSuccess(self.test)
        successOutput = self.stream.getvalue()
        self.assertEqual(successOutput, expectedFailureOutput)

    def test_subunitWithoutAddSkipInstalled(self):
        """
        Some versions of subunit don't have "addSkip". For these versions, we
        report skips as successes.
        """
        self.removeMethod(reporter.TestProtocolClient, 'addSkip')
        self.result.addSkip(self.test, 'reason')
        skipOutput = self.stream.getvalue()
        self.stream.truncate(0)
        self.stream.seek(0)
        self.result.addSuccess(self.test)
        successOutput = self.stream.getvalue()
        self.assertEqual(successOutput, skipOutput)

    def test_addExpectedFailurePassedThrough(self):
        """
        Some versions of subunit have "addExpectedFailure". For these
        versions, when we call 'addExpectedFailure' on the test result, we
        pass the error and test through to the subunit client.
        """
        addExpectedFailureCalls = []

        def addExpectedFailure(test, error):
            addExpectedFailureCalls.append((test, error))
        self.result._subunit.addExpectedFailure = addExpectedFailure
        try:
            1 / 0
        except ZeroDivisionError:
            exc_info = sys.exc_info()
            self.result.addExpectedFailure(self.test, exc_info, 'todo')
        self.assertEqual(addExpectedFailureCalls, [(self.test, exc_info)])

    def test_addSkipSendsSubunitAddSkip(self):
        """
        Some versions of subunit have "addSkip". For these versions, when we
        call 'addSkip' on the test result, we pass the test and reason through
        to the subunit client.
        """
        addSkipCalls = []

        def addSkip(test, reason):
            addSkipCalls.append((test, reason))
        self.result._subunit.addSkip = addSkip
        self.result.addSkip(self.test, 'reason')
        self.assertEqual(addSkipCalls, [(self.test, 'reason')])

    def test_doneDoesNothing(self):
        """
        The subunit reporter doesn't need to print out a summary -- the stream
        of results is everything. Thus, done() does nothing.
        """
        self.result.done()
        self.assertEqual(b'', self.stream.getvalue())

    def test_startTestSendsSubunitStartTest(self):
        """
        SubunitReporter.startTest() sends the subunit 'startTest' message.
        """
        self.assertForwardsToSubunit('startTest', self.test)

    def test_stopTestSendsSubunitStopTest(self):
        """
        SubunitReporter.stopTest() sends the subunit 'stopTest' message.
        """
        self.assertForwardsToSubunit('stopTest', self.test)

    def test_addSuccessSendsSubunitAddSuccess(self):
        """
        SubunitReporter.addSuccess() sends the subunit 'addSuccess' message.
        """
        self.assertForwardsToSubunit('addSuccess', self.test)

    def test_addErrorSendsSubunitAddError(self):
        """
        SubunitReporter.addError() sends the subunit 'addError' message.
        """
        try:
            1 / 0
        except ZeroDivisionError:
            error = sys.exc_info()
        self.assertForwardsToSubunit('addError', self.test, error)

    def test_addFailureSendsSubunitAddFailure(self):
        """
        SubunitReporter.addFailure() sends the subunit 'addFailure' message.
        """
        try:
            self.fail('hello')
        except self.failureException:
            failure = sys.exc_info()
        self.assertForwardsToSubunit('addFailure', self.test, failure)

    def test_addUnexpectedSuccessSendsSubunitAddSuccess(self):
        """
        SubunitReporter.addFailure() sends the subunit 'addSuccess' message,
        since subunit doesn't model unexpected success.
        """
        stream = BytesIO()
        subunitClient = reporter.TestProtocolClient(stream)
        subunitClient.addSuccess(self.test)
        subunitOutput = stream.getvalue()
        self.result.addUnexpectedSuccess(self.test)
        self.assertEqual(subunitOutput, self.stream.getvalue())

    def test_loadTimeErrors(self):
        """
        Load-time errors are reported like normal errors.
        """
        test = runner.TestLoader().loadByName('doesntexist')
        test.run(self.result)
        output = self.stream.getvalue()
        self.assertIn(b'doesntexist', output)