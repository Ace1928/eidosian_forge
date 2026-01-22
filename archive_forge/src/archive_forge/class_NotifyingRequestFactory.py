import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
class NotifyingRequestFactory:
    """
    A L{http.Request} factory that calls L{http.Request.notifyFinish} on all
    L{http.Request} objects before it returns them, and squirrels the resulting
    L{defer.Deferred} away on the class for later use. This is done as early
    as possible to ensure that we always see the result.
    """

    def __init__(self, wrappedFactory):
        self.results = []
        self._wrappedFactory = wrappedFactory
        for interface in providedBy(self._wrappedFactory):
            directlyProvides(self, interface)

    def __call__(self, *args, **kwargs):
        req = self._wrappedFactory(*args, **kwargs)
        self.results.append(req.notifyFinish())
        return _IDeprecatedHTTPChannelToRequestInterfaceProxy(req)