import os
import pdb
import sys
import unittest as pyunit
from io import StringIO
from typing import List
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted import plugin
from twisted.internet import defer
from twisted.plugins import twisted_trial
from twisted.python import failure, log, reflect
from twisted.python.filepath import FilePath
from twisted.python.reflect import namedAny
from twisted.scripts import trial
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import _ForceGarbageCollectionDecorator
from twisted.trial.itrial import IReporter, ITestCase
class UntilFailureTests(unittest.SynchronousTestCase):

    class FailAfter(pyunit.TestCase):
        """
        A test case that fails when run 3 times in a row.
        """
        count: List[None] = []

        def test_foo(self):
            self.count.append(None)
            if len(self.count) == 3:
                self.fail('Count reached 3')

    def setUp(self):
        UntilFailureTests.FailAfter.count = []
        self.test = UntilFailureTests.FailAfter('test_foo')
        self.stream = StringIO()
        self.runner = runner.TrialRunner(reporter.Reporter, stream=self.stream)

    def test_runUntilFailure(self):
        """
        Test that the runUntilFailure method of the runner actually fail after
        a few runs.
        """
        result = self.runner.runUntilFailure(self.test)
        self.assertEqual(result.testsRun, 1)
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(self._getFailures(result), 1)

    def _getFailures(self, result):
        """
        Get the number of failures that were reported to a result.
        """
        return len(result.failures)

    def test_runUntilFailureDecorate(self):
        """
        C{runUntilFailure} doesn't decorate the tests uselessly: it does it one
        time when run starts, but not at each turn.
        """
        decorated = []

        def decorate(test, interface):
            decorated.append((test, interface))
            return test
        self.patch(unittest, 'decorate', decorate)
        result = self.runner.runUntilFailure(self.test)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(decorated), 1)
        self.assertEqual(decorated, [(self.test, ITestCase)])

    def test_runUntilFailureForceGCDecorate(self):
        """
        C{runUntilFailure} applies the force-gc decoration after the standard
        L{ITestCase} decoration, but only one time.
        """
        decorated = []

        def decorate(test, interface):
            decorated.append((test, interface))
            return test
        self.patch(unittest, 'decorate', decorate)
        self.runner._forceGarbageCollection = True
        result = self.runner.runUntilFailure(self.test)
        self.assertEqual(result.testsRun, 1)
        self.assertEqual(len(decorated), 2)
        self.assertEqual(decorated, [(self.test, ITestCase), (self.test, _ForceGarbageCollectionDecorator)])