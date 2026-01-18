from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_rportValue(self):
    """
        An rport numeric setting should insert the parameter with the number
        value given.
        """
    v = sip.Via('foo.bar', rport=1)
    self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport=1')
    self.assertFalse(v.rportRequested)
    self.assertEqual(v.rportValue, 1)
    self.assertEqual(v.rport, 1)