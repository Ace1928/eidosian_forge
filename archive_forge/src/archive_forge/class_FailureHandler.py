from zope.interface import implementer
from twisted.internet import defer, interfaces
from twisted.names import common, dns, error
class FailureHandler:

    def __init__(self, resolver, query, timeout):
        self.resolver = resolver
        self.query = query
        self.timeout = timeout

    def __call__(self, failure):
        failure.trap(dns.DomainError, defer.TimeoutError, NotImplementedError)
        return self.resolver(self.query, self.timeout)