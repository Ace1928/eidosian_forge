from twisted.words.protocols.jabber import error, sasl, xmlstream
from twisted.words.protocols.jabber.jid import JID
from twisted.words.xish import domish, utility, xpath
def _ebAuthQuery(self, failure):
    failure.trap(error.StanzaError)
    e = failure.value
    if e.condition == 'not-authorized':
        self.xmlstream.dispatch(e.stanza, self.INVALID_USER_EVENT)
    else:
        self.xmlstream.dispatch(e.stanza, self.AUTH_FAILED_EVENT)
    return failure