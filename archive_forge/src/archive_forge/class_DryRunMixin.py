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
class DryRunMixin:
    """
    Mixin for testing that 'dry run' mode works with various
    L{pyunit.TestCase} subclasses.
    """

    def setUp(self):
        self.log = []
        self.stream = StringIO()
        self.runner = runner.TrialRunner(CapturingReporter, runner.TrialRunner.DRY_RUN, stream=self.stream)
        self.makeTestFixtures()

    def makeTestFixtures(self):
        """
        Set C{self.test} and C{self.suite}, where C{self.suite} is an empty
        TestSuite.
        """

    def test_empty(self):
        """
        If there are no tests, the reporter should not receive any events to
        report.
        """
        result = self.runner.run(runner.TestSuite())
        self.assertEqual(result._calls, [])

    def test_singleCaseReporting(self):
        """
        If we are running a single test, check the reporter starts, passes and
        then stops the test during a dry run.
        """
        result = self.runner.run(self.test)
        self.assertEqual(result._calls, ['startTest', 'addSuccess', 'stopTest'])

    def test_testsNotRun(self):
        """
        When we are doing a dry run, the tests should not actually be run.
        """
        self.runner.run(self.test)
        self.assertEqual(self.log, [])