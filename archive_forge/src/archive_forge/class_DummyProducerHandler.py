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
class DummyProducerHandler(http.Request):
    """
    An HTTP request handler that registers a dummy producer to serve the body.

    The owner must call C{finish} to complete the response.
    """

    def process(self):
        self.setResponseCode(200)
        self.registerProducer(DummyProducer(), True)