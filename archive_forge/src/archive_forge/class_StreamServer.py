from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
@implementer(IStreamServerEndpoint)
class StreamServer(EndpointBase):

    def listen(self, protocolFactory=None):
        pass