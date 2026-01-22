from typing import Iterable, cast
from zope.interface import Attribute, Interface, implementer
from twisted.internet.interfaces import IReactorCore
from twisted.plugin import IPlugin, getPlugins
from twisted.python.reflect import namedAny
@implementer(IPlugin, IReactorInstaller)
class Reactor:
    """
    @ivar moduleName: The fully-qualified Python name of the module of which
    the install callable is an attribute.
    """

    def __init__(self, shortName: str, moduleName: str, description: str):
        self.shortName = shortName
        self.moduleName = moduleName
        self.description = description

    def install(self) -> None:
        namedAny(self.moduleName).install()