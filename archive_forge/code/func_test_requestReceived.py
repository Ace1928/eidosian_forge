from __future__ import annotations
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.trial.unittest import TestCase
def test_requestReceived(self) -> None:
    """
        Test that requestReceived handles requests by dispatching them to
        request_* methods.
        """
    self.channel.request_test_method = lambda data: data == b''
    self.assertTrue(self.channel.requestReceived(b'test-method', b''))
    self.assertFalse(self.channel.requestReceived(b'test-method', b'a'))
    self.assertFalse(self.channel.requestReceived(b'bad-method', b''))