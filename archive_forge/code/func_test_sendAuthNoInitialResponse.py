from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_sendAuthNoInitialResponse(self):
    """
        Test starting authentication without an initial response.
        """
    self.init.initialResponse = None
    self.init.start()
    auth = self.output[0]
    self.assertEqual('', str(auth))