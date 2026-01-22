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
class TreeReporterTests(unittest.SynchronousTestCase):

    def setUp(self):
        self.test = sample.FooTest('test_foo')
        self.stream = StringIO()
        self.result = reporter.TreeReporter(self.stream)
        self.result._colorizer = MockColorizer(self.stream)
        self.log = self.result._colorizer.log

    def makeError(self):
        try:
            1 / 0
        except ZeroDivisionError:
            f = Failure()
        return f

    def test_summaryColoredSuccess(self):
        """
        The summary in case of success should have a good count of successes
        and be colored properly.
        """
        self.result.addSuccess(self.test)
        self.result.done()
        self.assertEqual(self.log[1], (self.result.SUCCESS, 'PASSED'))
        self.assertEqual(self.stream.getvalue().splitlines()[-1].strip(), '(successes=1)')

    def test_summaryColoredFailure(self):
        """
        The summary in case of failure should have a good count of errors
        and be colored properly.
        """
        try:
            raise RuntimeError('foo')
        except RuntimeError:
            self.result.addError(self, sys.exc_info())
        self.result.done()
        self.assertEqual(self.log[1], (self.result.FAILURE, 'FAILED'))
        self.assertEqual(self.stream.getvalue().splitlines()[-1].strip(), '(errors=1)')

    def test_getPrelude(self):
        """
        The tree needs to get the segments of the test ID that correspond
        to the module and class that it belongs to.
        """
        self.assertEqual(['foo.bar', 'baz'], self.result._getPreludeSegments('foo.bar.baz.qux'))
        self.assertEqual(['foo', 'bar'], self.result._getPreludeSegments('foo.bar.baz'))
        self.assertEqual(['foo'], self.result._getPreludeSegments('foo.bar'))
        self.assertEqual([], self.result._getPreludeSegments('foo'))

    def test_groupResults(self):
        """
        If two different tests have the same error, L{Reporter._groupResults}
        includes them together in one of the tuples in the list it returns.
        """
        try:
            raise RuntimeError('foo')
        except RuntimeError:
            self.result.addError(self, sys.exc_info())
            self.result.addError(self.test, sys.exc_info())
        try:
            raise RuntimeError('bar')
        except RuntimeError:
            extra = sample.FooTest('test_bar')
            self.result.addError(extra, sys.exc_info())
        self.result.done()
        grouped = self.result._groupResults(self.result.errors, self.result._formatFailureTraceback)
        self.assertEqual(grouped[0][1], [self, self.test])
        self.assertEqual(grouped[1][1], [extra])

    def test_printResults(self):
        """
        L{Reporter._printResults} uses the results list and formatter callable
        passed to it to produce groups of results to write to its output
        stream.
        """

        def formatter(n):
            return str(n) + '\n'
        first = sample.FooTest('test_foo')
        second = sample.FooTest('test_bar')
        third = sample.PyunitTest('test_foo')
        self.result._printResults('FOO', [(first, 1), (second, 1), (third, 2)], formatter)
        self.assertEqual(self.stream.getvalue(), '%(double separator)s\nFOO\n1\n\n%(first)s\n%(second)s\n%(double separator)s\nFOO\n2\n\n%(third)s\n' % {'double separator': self.result._doubleSeparator, 'first': first.id(), 'second': second.id(), 'third': third.id()})