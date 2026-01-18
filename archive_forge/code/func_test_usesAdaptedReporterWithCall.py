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
def test_usesAdaptedReporterWithCall(self):
    """
        For decorated tests, C{__call__} uses a result adapter that preserves
        the test decoration for calls to C{addError}, C{startTest} and the
        like.

        See L{reporter._AdaptedReporter}.
        """
    test = self.TestCase()
    decoratedTest = unittest.TestDecorator(test)
    from twisted.trial.test.test_reporter import LoggingReporter
    result = LoggingReporter()
    decoratedTest(result)
    self.assertTestsEqual(result.test, decoratedTest)