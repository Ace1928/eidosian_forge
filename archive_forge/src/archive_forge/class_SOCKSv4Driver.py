import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class SOCKSv4Driver(socks.SOCKSv4):
    driver_outgoing = None
    driver_listen = None

    def connectClass(self, host, port, klass, *args):
        proto = klass(*args)
        proto.transport = StringTCPTransport()
        proto.transport.peer = address.IPv4Address('TCP', host, port)
        proto.connectionMade()
        self.driver_outgoing = proto
        return defer.succeed(proto)

    def listenClass(self, port, klass, *args):
        factory = klass(*args)
        self.driver_listen = factory
        if port == 0:
            port = 1234
        return defer.succeed(('6.7.8.9', port))