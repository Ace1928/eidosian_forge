import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
def test_oversizedString(self):
    data = b'\x02\x01\x01\x01\x01\x82'
    self.assertRaises(banana.BananaError, self.feed, data)