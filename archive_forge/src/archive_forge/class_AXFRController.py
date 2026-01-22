import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
class AXFRController:
    timeoutCall = None

    def __init__(self, name, deferred):
        self.name = name
        self.deferred = deferred
        self.soa = None
        self.records = []
        self.pending = [(deferred,)]

    def connectionMade(self, protocol):
        message = dns.Message(protocol.pickID(), recDes=0)
        message.queries = [dns.Query(self.name, dns.AXFR, dns.IN)]
        protocol.writeMessage(message)

    def connectionLost(self, protocol):
        pass

    def messageReceived(self, message, protocol):
        self.records.extend(message.answers)
        if not self.records:
            return
        if not self.soa:
            if self.records[0].type == dns.SOA:
                self.soa = self.records[0]
        if len(self.records) > 1 and self.records[-1].type == dns.SOA:
            if self.timeoutCall is not None:
                self.timeoutCall.cancel()
                self.timeoutCall = None
            if self.deferred is not None:
                self.deferred.callback(self.records)
                self.deferred = None