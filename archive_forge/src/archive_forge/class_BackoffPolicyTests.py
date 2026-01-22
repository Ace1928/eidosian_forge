import copy
import os
import pickle
from io import StringIO
from unittest import skipIf
from twisted.application import app, internet, reactors, service
from twisted.application.internet import backoffPolicy
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet.testing import MemoryReactor
from twisted.persisted import sob
from twisted.plugins import twisted_reactors
from twisted.protocols import basic, wire
from twisted.python import usage
from twisted.python.runtime import platformType
from twisted.python.test.modules_helpers import TwistedModulesMixin
from twisted.trial.unittest import SkipTest, TestCase
class BackoffPolicyTests(TestCase):
    """
    Tests of L{twisted.application.internet.backoffPolicy}
    """

    def test_calculates_correct_values(self):
        """
        Test that L{backoffPolicy()} calculates expected values
        """
        pol = backoffPolicy(1.0, 60.0, 1.5, jitter=lambda: 1)
        self.assertAlmostEqual(pol(0), 2)
        self.assertAlmostEqual(pol(1), 2.5)
        self.assertAlmostEqual(pol(10), 58.6650390625)
        self.assertEqual(pol(20), 61)
        self.assertEqual(pol(100), 61)

    def test_does_not_overflow_on_high_attempts(self):
        """
        L{backoffPolicy()} does not fail for large values of the attempt
        parameter. In previous versions, this test failed when attempt was
        larger than 1750.

        See https://twistedmatrix.com/trac/ticket/9476
        """
        pol = backoffPolicy(1.0, 60.0, 1.5, jitter=lambda: 1)
        self.assertEqual(pol(1751), 61)
        self.assertEqual(pol(1000000), 61)

    def test_does_not_overflow_with_large_factor_value(self):
        """
        Even with unusual parameters, any L{OverflowError} within
        L{backoffPolicy()} will be caught and L{maxDelay} will be returned
        instead
        """
        pol = backoffPolicy(1.0, 60.0, 10000000000.0, jitter=lambda: 1)
        self.assertEqual(pol(1751), 61)