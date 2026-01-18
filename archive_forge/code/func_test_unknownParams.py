from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
from zope.interface import implementer
def test_unknownParams(self):
    """
        Parsing and serializing Via headers with unknown parameters should work.
        """
    s = 'SIP/2.0/UDP example.com:5060;branch=a12345b;bogus;pie=delicious'
    v = sip.parseViaHeader(s)
    self.assertEqual(v.toString(), s)