from zope.interface.declarations import implementer
from twisted.internet.interfaces import (
from twisted.plugin import IPlugin
@implementer(IPlugin)
class PluginBase:

    def __init__(self, pfx):
        self.prefix = pfx