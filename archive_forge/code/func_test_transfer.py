import copy
import operator
import socket
from functools import partial, reduce
from io import BytesIO
from struct import pack
from twisted.internet import defer, error, reactor
from twisted.internet.defer import succeed
from twisted.internet.testing import (
from twisted.names import authority, client, common, dns, server
from twisted.names.client import Resolver
from twisted.names.dns import SOA, Message, Query, Record_A, Record_SOA, RRHeader
from twisted.names.error import DomainError
from twisted.names.secondary import SecondaryAuthority, SecondaryAuthorityService
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.trial import unittest
def test_transfer(self):
    """
        An attempt is made to transfer the zone for the domain the
        L{SecondaryAuthority} was constructed with from the server address it
        was constructed with when L{SecondaryAuthority.transfer} is called.
        """
    secondary = SecondaryAuthority.fromServerAddressAndDomain(('192.168.1.2', 1234), 'example.com')
    secondary._reactor = reactor = MemoryReactorClock()
    secondary.transfer()
    host, port, factory, timeout, bindAddress = reactor.tcpClients.pop(0)
    self.assertEqual(host, '192.168.1.2')
    self.assertEqual(port, 1234)
    proto = factory.buildProtocol((host, port))
    transport = StringTransport()
    proto.makeConnection(transport)
    msg = Message()
    msg.decode(BytesIO(transport.value()[2:]))
    self.assertEqual([dns.Query('example.com', dns.AXFR, dns.IN)], msg.queries)