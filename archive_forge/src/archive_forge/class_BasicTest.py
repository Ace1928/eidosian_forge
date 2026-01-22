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
class BasicTest(unittest.SynchronousTestCase):
    """
        Mock test to run.
        """

    def setUp(self):
        """
            Mock setUp
            """
        self._log('setUp')

    def test_foo(self):
        """
            Mock test case
            """
        self._log('test')

    def tearDown(self):
        """
            Mock tear tearDown
            """
        self._log('tearDown')