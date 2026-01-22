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
def assertTestsEqual(self, observed, expected):
    """
        Assert that the given decorated tests are equal.
        """
    self.assertEqual(observed.__class__, expected.__class__, 'Different class')
    observedOriginal = getattr(observed, '_originalTest', None)
    expectedOriginal = getattr(expected, '_originalTest', None)
    self.assertIdentical(observedOriginal, expectedOriginal)
    if observedOriginal is expectedOriginal is None:
        self.assertIdentical(observed, expected)