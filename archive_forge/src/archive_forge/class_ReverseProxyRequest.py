from urllib.parse import quote as urlquote, urlparse, urlunparse
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory
from twisted.web.http import _QUEUED_SENTINEL, HTTPChannel, HTTPClient, Request
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET
class ReverseProxyRequest(Request):
    """
    Used by ReverseProxy to implement a simple reverse proxy.

    @ivar proxyClientFactoryClass: a proxy client factory class, used to create
        new connections.
    @type proxyClientFactoryClass: L{ClientFactory}

    @ivar reactor: the reactor used to create connections.
    @type reactor: object providing L{twisted.internet.interfaces.IReactorTCP}
    """
    proxyClientFactoryClass = ProxyClientFactory

    def __init__(self, channel, queued=_QUEUED_SENTINEL, reactor=reactor):
        Request.__init__(self, channel, queued)
        self.reactor = reactor

    def process(self):
        """
        Handle this request by connecting to the proxied server and forwarding
        it there, then forwarding the response back as the response to this
        request.
        """
        self.requestHeaders.setRawHeaders(b'host', [self.factory.host.encode('ascii')])
        clientFactory = self.proxyClientFactoryClass(self.method, self.uri, self.clientproto, self.getAllHeaders(), self.content.read(), self)
        self.reactor.connectTCP(self.factory.host, self.factory.port, clientFactory)