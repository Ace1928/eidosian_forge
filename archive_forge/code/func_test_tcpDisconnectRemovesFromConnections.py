import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_tcpDisconnectRemovesFromConnections(self):
    """
        When a TCP DNS protocol associated with a Resolver disconnects, it is
        removed from the Resolver's connection list.
        """
    resolver = client.Resolver(servers=[('example.com', 53)])
    protocol = resolver.factory.buildProtocol(None)
    protocol.makeConnection(None)
    self.assertIn(protocol, resolver.connections)
    protocol.connectionLost(None)
    self.assertNotIn(protocol, resolver.connections)