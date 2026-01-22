from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
class EndpointBase:

    def __init__(self, parser, args, kwargs):
        self.parser = parser
        self.args = args
        self.kwargs = kwargs