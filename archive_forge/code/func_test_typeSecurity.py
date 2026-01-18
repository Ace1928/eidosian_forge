import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_typeSecurity(self):
    """
        Test for type-level security of serialization.
        """
    taster = jelly.SecurityOptions()
    dct = jelly.jelly({})
    self.assertRaises(jelly.InsecureJelly, jelly.unjelly, dct, taster)