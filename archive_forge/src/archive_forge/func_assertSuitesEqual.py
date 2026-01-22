import gc
import sys
import unittest as pyunit
import weakref
from io import StringIO
from twisted.internet import defer, reactor
from twisted.python.compat import _PYPY
from twisted.python.reflect import namedAny
from twisted.trial import reporter, runner, unittest, util
from twisted.trial._asyncrunner import (
from twisted.trial.test import erroneous
from twisted.trial.test.test_suppression import SuppressionMixin
def assertSuitesEqual(self, observed, expected):
    """
        Assert that the given test suites with decorated tests are equal.
        """
    self.assertEqual(observed.__class__, expected.__class__, 'Different class')
    self.assertEqual(len(observed._tests), len(expected._tests), 'Different number of tests.')
    for observedTest, expectedTest in zip(observed._tests, expected._tests):
        if getattr(observedTest, '_tests', None) is not None:
            self.assertSuitesEqual(observedTest, expectedTest)
        else:
            self.assertTestsEqual(observedTest, expectedTest)