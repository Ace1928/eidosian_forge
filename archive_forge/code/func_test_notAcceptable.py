from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_notAcceptable(self):
    """
        Test using an unacceptable SASL authentication mechanism.
        """
    self.authenticator.jid = jid.JID('test@example.com')
    self.authenticator.password = 'secret'
    self.assertRaises(sasl.SASLNoAcceptableMechanism, self._setMechanism, 'SOMETHING_UNACCEPTABLE')