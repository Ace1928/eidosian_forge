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
class FakeDNSDatagramProtocol:

    def __init__(self):
        self.queries = []
        self.transport = StubPort()

    def query(self, address, queries, timeout=10, id=None):
        self.queries.append((address, queries, timeout, id))
        return defer.fail(error.DNSQueryTimeoutError(queries))

    def removeResend(self, id):
        pass