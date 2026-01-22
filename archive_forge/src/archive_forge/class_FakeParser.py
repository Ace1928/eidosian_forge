from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
@implementer(IStreamServerEndpointStringParser)
class FakeParser(PluginBase):

    def parseStreamServer(self, *a, **kw):
        return StreamServer(self, a, kw)