import sys
from functools import partial
from io import BytesIO
from twisted.internet import main, protocol
from twisted.internet.testing import StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.spread import banana
from twisted.trial.unittest import TestCase
class GlobalCoderTests(TestCase):
    """
    Tests for the free functions L{banana.encode} and L{banana.decode}.
    """

    def test_statelessDecode(self):
        """
        Calls to L{banana.decode} are independent of each other.
        """
        undecodable = b'\x7f' * 65 + b'\x85'
        self.assertRaises(banana.BananaError, banana.decode, undecodable)
        decodable = b'\x01\x81'
        self.assertEqual(banana.decode(decodable), 1)