from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer
from twisted.persisted import sob
from twisted.plugin import IPlugin
from twisted.python import components
from twisted.python.reflect import namedAny
@implementer(IServiceCollection)
class MultiService(Service):
    """
    Straightforward Service Container.

    Hold a collection of services, and manage them in a simplistic
    way. No service will wait for another, but this object itself
    will not finish shutting down until all of its child services
    will finish.
    """

    def __init__(self):
        self.services = []
        self.namedServices = {}
        self.parent = None

    def privilegedStartService(self):
        Service.privilegedStartService(self)
        for service in self:
            service.privilegedStartService()

    def startService(self):
        Service.startService(self)
        for service in self:
            service.startService()

    def stopService(self):
        Service.stopService(self)
        l = []
        services = list(self)
        services.reverse()
        for service in services:
            l.append(defer.maybeDeferred(service.stopService))
        return defer.DeferredList(l)

    def getServiceNamed(self, name):
        return self.namedServices[name]

    def __iter__(self):
        return iter(self.services)

    def addService(self, service):
        if service.name is not None:
            if service.name in self.namedServices:
                raise RuntimeError("cannot have two services with same name '%s'" % service.name)
            self.namedServices[service.name] = service
        self.services.append(service)
        if self.running:
            service.privilegedStartService()
            service.startService()

    def removeService(self, service):
        if service.name:
            del self.namedServices[service.name]
        self.services.remove(service)
        if self.running:
            return service.stopService()
        else:
            return None