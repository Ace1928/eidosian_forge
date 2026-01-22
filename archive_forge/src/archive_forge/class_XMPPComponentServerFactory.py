from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
class XMPPComponentServerFactory(xmlstream.XmlStreamServerFactory):
    """
    XMPP Component Server factory.

    This factory accepts XMPP external component connections and makes
    the router service route traffic for a component's bound domain
    to that component.

    @since: 8.2
    """
    logTraffic = False

    def __init__(self, router, secret='secret'):
        self.router = router
        self.secret = secret

        def authenticatorFactory():
            return ListenComponentAuthenticator(self.secret)
        xmlstream.XmlStreamServerFactory.__init__(self, authenticatorFactory)
        self.addBootstrap(xmlstream.STREAM_CONNECTED_EVENT, self.onConnectionMade)
        self.addBootstrap(xmlstream.STREAM_AUTHD_EVENT, self.onAuthenticated)
        self.serial = 0

    def onConnectionMade(self, xs):
        """
        Called when a component connection was made.

        This enables traffic debugging on incoming streams.
        """
        xs.serial = self.serial
        self.serial += 1

        def logDataIn(buf):
            log.msg('RECV (%d): %r' % (xs.serial, buf))

        def logDataOut(buf):
            log.msg('SEND (%d): %r' % (xs.serial, buf))
        if self.logTraffic:
            xs.rawDataInFn = logDataIn
            xs.rawDataOutFn = logDataOut
        xs.addObserver(xmlstream.STREAM_ERROR_EVENT, self.onError)

    def onAuthenticated(self, xs):
        """
        Called when a component has successfully authenticated.

        Add the component to the routing table and establish a handler
        for a closed connection.
        """
        destination = xs.thisEntity.host
        self.router.addRoute(destination, xs)
        xs.addObserver(xmlstream.STREAM_END_EVENT, self.onConnectionLost, 0, destination, xs)

    def onError(self, reason):
        log.err(reason, 'Stream Error')

    def onConnectionLost(self, destination, xs, reason):
        self.router.removeRoute(destination, xs)