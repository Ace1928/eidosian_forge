from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_onChallenge(self):
    """
        Test receiving a challenge message.
        """
    d = self.init.start()
    challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
    challenge.addContent('bXkgY2hhbGxlbmdl')
    self.init.onChallenge(challenge)
    self.assertEqual(b'my challenge', self.init.mechanism.challenge)
    self.init.onSuccess(None)
    return d