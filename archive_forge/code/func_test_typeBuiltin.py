import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_typeBuiltin(self):
    """
        Test that a builtin type can be jellied and unjellied to the original
        type.
        """
    t = [str]
    r = jelly.unjelly(jelly.jelly(t))
    self.assertEqual(t, r)