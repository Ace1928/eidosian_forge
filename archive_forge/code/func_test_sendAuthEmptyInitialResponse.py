from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_sendAuthEmptyInitialResponse(self):
    """
        Test starting authentication where the initial response is empty.
        """
    self.init.initialResponse = b''
    self.init.start()
    auth = self.output[0]
    self.assertEqual('=', str(auth))