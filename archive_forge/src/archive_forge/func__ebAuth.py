from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def _ebAuth(self, failure):
    failure.trap(error.StanzaError)
    self.xmlstream.dispatch(failure.value.stanza, self.AUTH_FAILED_EVENT)
    return failure