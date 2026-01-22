from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
def addService(self, service):
    if service.name is not None:
        if service.name in self.namedServices:
            raise RuntimeError("cannot have two services with same name '%s'" % service.name)
        self.namedServices[service.name] = service
    self.services.append(service)
    if self.running:
        service.privilegedStartService()
        service.startService()