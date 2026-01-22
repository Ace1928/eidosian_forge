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
class AsynchronousSkipMethodTests(SkipMethodsMixin, unittest.TestCase):
    """
    Tests for the reporting of skipping tests in the asynchronous case.

    See: L{twisted.trial.test.test_tests.SkipMethodsMixin}
    """
    Skipping = namedAny('twisted.trial.test.skipping.AsynchronousSkipping')
    SkippingSetUp = namedAny('twisted.trial.test.skipping.AsynchronousSkippingSetUp')
    DeprecatedReasonlessSkip = namedAny('twisted.trial.test.skipping.AsynchronousDeprecatedReasonlessSkip')