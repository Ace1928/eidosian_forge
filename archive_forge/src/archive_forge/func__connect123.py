import builtins
from io import StringIO
from zope.interface import Interface, implementedBy, implementer
from twisted.internet import address, defer, protocol, reactor, task
from twisted.internet.testing import StringTransport, StringTransportWithDisconnection
from twisted.protocols import policies
from twisted.trial import unittest
def _connect123(results):
    reactor.connectTCP('127.0.0.1', n, SillyFactory(c1))
    c1.dConnected.addCallback(lambda r: reactor.connectTCP('127.0.0.1', n, SillyFactory(c2)))
    c2.dConnected.addCallback(lambda r: reactor.connectTCP('127.0.0.1', n, SillyFactory(c3)))
    return c3.dDisconnected