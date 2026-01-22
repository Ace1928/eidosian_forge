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
class DestructiveTestSuiteTests(unittest.SynchronousTestCase):
    """
    Test for L{runner.DestructiveTestSuite}.
    """

    def test_basic(self):
        """
        Thes destructive test suite should run the tests normally.
        """
        called = []

        class MockTest(pyunit.TestCase):

            def test_foo(test):
                called.append(True)
        test = MockTest('test_foo')
        result = reporter.TestResult()
        suite = runner.DestructiveTestSuite([test])
        self.assertEqual(called, [])
        suite.run(result)
        self.assertEqual(called, [True])
        self.assertEqual(suite.countTestCases(), 0)

    def test_shouldStop(self):
        """
        Test the C{shouldStop} management: raising a C{KeyboardInterrupt} must
        interrupt the suite.
        """
        called = []

        class MockTest(unittest.TestCase):

            def test_foo1(test):
                called.append(1)

            def test_foo2(test):
                raise KeyboardInterrupt()

            def test_foo3(test):
                called.append(2)
        result = reporter.TestResult()
        loader = runner.TestLoader()
        loader.suiteFactory = runner.DestructiveTestSuite
        suite = loader.loadClass(MockTest)
        self.assertEqual(called, [])
        suite.run(result)
        self.assertEqual(called, [1])
        self.assertEqual(suite.countTestCases(), 1)

    def test_cleanup(self):
        """
        Checks that the test suite cleanups its tests during the run, so that
        it ends empty.
        """

        class MockTest(pyunit.TestCase):

            def test_foo(test):
                pass
        test = MockTest('test_foo')
        result = reporter.TestResult()
        suite = runner.DestructiveTestSuite([test])
        self.assertEqual(suite.countTestCases(), 1)
        suite.run(result)
        self.assertEqual(suite.countTestCases(), 0)