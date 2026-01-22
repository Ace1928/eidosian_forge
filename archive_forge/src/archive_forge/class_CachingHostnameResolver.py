from typing import Any
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache
@implementer(IHostnameResolver)
class CachingHostnameResolver:
    """
    Experimental caching resolver. Resolves IPv4 and IPv6 addresses,
    does not support setting a timeout value for DNS requests.
    """

    def __init__(self, reactor, cache_size):
        self.reactor = reactor
        self.original_resolver = reactor.nameResolver
        dnscache.limit = cache_size

    @classmethod
    def from_crawler(cls, crawler, reactor):
        if crawler.settings.getbool('DNSCACHE_ENABLED'):
            cache_size = crawler.settings.getint('DNSCACHE_SIZE')
        else:
            cache_size = 0
        return cls(reactor, cache_size)

    def install_on_reactor(self):
        self.reactor.installNameResolver(self)

    def resolveHostName(self, resolutionReceiver, hostName: str, portNumber=0, addressTypes=None, transportSemantics='TCP'):
        try:
            addresses = dnscache[hostName]
        except KeyError:
            return self.original_resolver.resolveHostName(_CachingResolutionReceiver(resolutionReceiver, hostName), hostName, portNumber, addressTypes, transportSemantics)
        else:
            resolutionReceiver.resolutionBegan(HostResolution(hostName))
            for addr in addresses:
                resolutionReceiver.addressResolved(addr)
            resolutionReceiver.resolutionComplete()
            return resolutionReceiver