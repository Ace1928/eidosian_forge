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
class AsynchronousSuppressionTests(SuppressionMixin, unittest.TestCase):
    """
    Tests for the warning suppression features of
    L{twisted.trial.unittest.TestCase}

    See L{twisted.trial.test.test_suppression.SuppressionMixin}
    """
    TestSetUpSuppression = namedAny('twisted.trial.test.suppression.AsynchronousTestSetUpSuppression')
    TestTearDownSuppression = namedAny('twisted.trial.test.suppression.AsynchronousTestTearDownSuppression')
    TestSuppression = namedAny('twisted.trial.test.suppression.AsynchronousTestSuppression')
    TestSuppression2 = namedAny('twisted.trial.test.suppression.AsynchronousTestSuppression2')