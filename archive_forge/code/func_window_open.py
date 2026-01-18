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
def window_open():
    a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
    self.assertTrue(stream._producerProducing)
    self.assertEqual(request.producer.events, ['pause', 'resume'])
    request.unregisterProducer()
    request.finish()