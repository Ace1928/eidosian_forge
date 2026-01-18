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
def test_patchRestoredAfterRun(self):
    """
        Any monkey patches introduced by a test using C{patch()} are reverted
        after the test has run.
        """
    self.test.patch(self, 'objectToPatch', self.patchedValue)
    self.test.run(reporter.Reporter())
    self.assertEqual(self.objectToPatch, self.originalValue)