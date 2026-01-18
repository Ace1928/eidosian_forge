from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_rport(self):
    """
        An rport setting of None should insert the parameter with no value.
        """
    v = sip.Via('foo.bar', rport=None)
    self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
    self.assertTrue(v.rportRequested)
    self.assertIsNone(v.rportValue)