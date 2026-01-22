from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
class ListenComponentAuthenticator(xmlstream.ListenAuthenticator):
    """
    Authenticator for accepting components.

    @since: 8.2
    @ivar secret: The shared secret used to authorized incoming component
                  connections.
    @type secret: C{unicode}.
    """
    namespace = NS_COMPONENT_ACCEPT

    def __init__(self, secret):
        self.secret = secret
        xmlstream.ListenAuthenticator.__init__(self)

    def associateWithStream(self, xs):
        """
        Associate the authenticator with a stream.

        This sets the stream's version to 0.0, because the XEP-0114 component
        protocol was not designed for XMPP 1.0.
        """
        xs.version = (0, 0)
        xmlstream.ListenAuthenticator.associateWithStream(self, xs)

    def streamStarted(self, rootElement):
        """
        Called by the stream when it has started.

        This examines the default namespace of the incoming stream and whether
        there is a requested hostname for the component. Then it generates a
        stream identifier, sends a response header and adds an observer for
        the first incoming element, triggering L{onElement}.
        """
        xmlstream.ListenAuthenticator.streamStarted(self, rootElement)
        if rootElement.defaultUri != self.namespace:
            exc = error.StreamError('invalid-namespace')
            self.xmlstream.sendStreamError(exc)
            return
        if not self.xmlstream.thisEntity:
            exc = error.StreamError('improper-addressing')
            self.xmlstream.sendStreamError(exc)
            return
        self.xmlstream.sendHeader()
        self.xmlstream.addOnetimeObserver('/*', self.onElement)

    def onElement(self, element):
        """
        Called on incoming XML Stanzas.

        The very first element received should be a request for handshake.
        Otherwise, the stream is dropped with a 'not-authorized' error. If a
        handshake request was received, the hash is extracted and passed to
        L{onHandshake}.
        """
        if (element.uri, element.name) == (self.namespace, 'handshake'):
            self.onHandshake(str(element))
        else:
            exc = error.StreamError('not-authorized')
            self.xmlstream.sendStreamError(exc)

    def onHandshake(self, handshake):
        """
        Called upon receiving the handshake request.

        This checks that the given hash in C{handshake} is equal to a
        calculated hash, responding with a handshake reply or a stream error.
        If the handshake was ok, the stream is authorized, and  XML Stanzas may
        be exchanged.
        """
        calculatedHash = xmlstream.hashPassword(self.xmlstream.sid, str(self.secret))
        if handshake != calculatedHash:
            exc = error.StreamError('not-authorized', text='Invalid hash')
            self.xmlstream.sendStreamError(exc)
        else:
            self.xmlstream.send('<handshake/>')
            self.xmlstream.dispatch(self.xmlstream, xmlstream.STREAM_AUTHD_EVENT)