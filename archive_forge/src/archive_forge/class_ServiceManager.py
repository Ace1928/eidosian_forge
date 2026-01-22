from zope.interface import implementer
from twisted.application import service
from twisted.internet import defer
from twisted.python import log
from twisted.words.protocols.jabber import error, ijabber, jstrports, xmlstream
from twisted.words.protocols.jabber.jid import internJID as JID
from twisted.words.xish import domish
class ServiceManager(service.MultiService):
    """
    Business logic for a managed component connection to a Jabber router.

    This service maintains a single connection to a Jabber router and provides
    facilities for packet routing and transmission. Business logic modules are
    services implementing L{ijabber.IService} (like subclasses of L{Service}),
    and added as sub-service.
    """

    def __init__(self, jid, password):
        service.MultiService.__init__(self)
        self.jabberId = jid
        self.xmlstream = None
        self._packetQueue = []
        self._xsFactory = componentFactory(self.jabberId, password)
        self._xsFactory.addBootstrap(xmlstream.STREAM_CONNECTED_EVENT, self._connected)
        self._xsFactory.addBootstrap(xmlstream.STREAM_AUTHD_EVENT, self._authd)
        self._xsFactory.addBootstrap(xmlstream.STREAM_END_EVENT, self._disconnected)
        self.addBootstrap = self._xsFactory.addBootstrap
        self.removeBootstrap = self._xsFactory.removeBootstrap

    def getFactory(self):
        return self._xsFactory

    def _connected(self, xs):
        self.xmlstream = xs
        for c in self:
            if ijabber.IService.providedBy(c):
                c.transportConnected(xs)

    def _authd(self, xs):
        for p in self._packetQueue:
            self.xmlstream.send(p)
        self._packetQueue = []
        for c in self:
            if ijabber.IService.providedBy(c):
                c.componentConnected(xs)

    def _disconnected(self, _):
        self.xmlstream = None
        for c in self:
            if ijabber.IService.providedBy(c):
                c.componentDisconnected()

    def send(self, obj):
        """
        Send data over the XML stream.

        When there is no established XML stream, the data is queued and sent
        out when a new XML stream has been established and initialized.

        @param obj: data to be sent over the XML stream. This is usually an
        object providing L{domish.IElement}, or serialized XML. See
        L{xmlstream.XmlStream} for details.
        """
        if self.xmlstream != None:
            self.xmlstream.send(obj)
        else:
            self._packetQueue.append(obj)