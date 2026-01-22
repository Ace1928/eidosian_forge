import socket
import struct
from twisted.internet import address, defer
from twisted.internet.error import DNSLookupError
from twisted.protocols import socks
from twisted.python.compat import iterbytes
from twisted.test import proto_helpers
from twisted.trial import unittest
class FakeResolverReactor:
    """
    Bare-bones reactor with deterministic behavior for the resolve method.
    """

    def __init__(self, names):
        """
        @type names: L{dict} containing L{str} keys and L{str} values.
        @param names: A hostname to IP address mapping. The IP addresses are
            stringified dotted quads.
        """
        self.names = names

    def resolve(self, hostname):
        """
        Resolve a hostname by looking it up in the C{names} dictionary.
        """
        try:
            return defer.succeed(self.names[hostname])
        except KeyError:
            return defer.fail(DNSLookupError("FakeResolverReactor couldn't find " + hostname.decode('utf-8')))