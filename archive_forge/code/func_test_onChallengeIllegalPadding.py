from zope.interface import implementer
from twisted.internet import defer
from twisted.trial import unittest
from twisted.words.protocols.jabber import jid, sasl, sasl_mechanisms, xmlstream
from twisted.words.xish import domish
def test_onChallengeIllegalPadding(self):
    """
        Test receiving a challenge message with illegal padding.
        """
    d = self.init.start()
    challenge = domish.Element((NS_XMPP_SASL, 'challenge'))
    challenge.addContent('bXkg=Y2hhbGxlbmdl')
    self.init.onChallenge(challenge)
    self.assertFailure(d, sasl.SASLIncorrectEncodingError)
    return d