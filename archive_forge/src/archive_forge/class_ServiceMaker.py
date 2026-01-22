from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
@implementer(IPlugin, IServiceMaker)
class ServiceMaker:
    """
    Utility class to simplify the definition of L{IServiceMaker} plugins.
    """

    def __init__(self, name, module, description, tapname):
        self.name = name
        self.module = module
        self.description = description
        self.tapname = tapname

    @property
    def options(self):
        return namedAny(self.module).Options

    @property
    def makeService(self):
        return namedAny(self.module).makeService