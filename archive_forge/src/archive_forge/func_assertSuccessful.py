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
def assertSuccessful(self, test, result):
    """
        Utility function -- assert there is one success and the state is
        plausible
        """
    self.assertEqual(result.successes, 1)
    self.assertEqual(result.failures, [])
    self.assertEqual(result.errors, [])
    self.assertEqual(result.expectedFailures, [])
    self.assertEqual(result.unexpectedSuccesses, [])
    self.assertEqual(result.skips, [])