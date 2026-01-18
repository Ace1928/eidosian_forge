from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_onChallengeEmpty(self):
    """
        Test receiving an empty challenge message.
        """
    d = self.init.start()
    challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
    self.init.onChallenge(challenge)
    self.assertEqual(b'', self.init.mechanism.challenge)
    self.init.onSuccess(None)
    return d