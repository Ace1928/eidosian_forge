import datetime
import decimal
from twisted.internet.testing import StringTransport
from twisted.spread import banana, jelly, pb
from twisted.trial import unittest
from twisted.trial.unittest import TestCase
def test_setSecurity(self):
    """
        By default, C{set} objects should be allowed by
        L{jelly.SecurityOptions}. If not allowed, L{jelly.unjelly} should raise
        L{jelly.InsecureJelly} when trying to unjelly it.
        """
    inputList = [{1, 2, 3}]
    self._testSecurity(inputList, b'set')