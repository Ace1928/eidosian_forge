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
class AXFRTests(unittest.TestCase):

    def setUp(self):
        self.results = None
        self.d = defer.Deferred()
        self.d.addCallback(self._gotResults)
        self.controller = client.AXFRController('fooby.com', self.d)
        self.soa = dns.RRHeader(name='fooby.com', type=dns.SOA, cls=dns.IN, ttl=86400, auth=False, payload=dns.Record_SOA(mname='fooby.com', rname='hooj.fooby.com', serial=100, refresh=200, retry=300, expire=400, minimum=500, ttl=600))
        self.records = [self.soa, dns.RRHeader(name='fooby.com', type=dns.NS, cls=dns.IN, ttl=700, auth=False, payload=dns.Record_NS(name='ns.twistedmatrix.com', ttl=700)), dns.RRHeader(name='fooby.com', type=dns.MX, cls=dns.IN, ttl=700, auth=False, payload=dns.Record_MX(preference=10, exchange='mail.mv3d.com', ttl=700)), dns.RRHeader(name='fooby.com', type=dns.A, cls=dns.IN, ttl=700, auth=False, payload=dns.Record_A(address='64.123.27.105', ttl=700)), self.soa]

    def _makeMessage(self):
        return dns.Message(id=999, answer=1, opCode=0, recDes=0, recAv=1, auth=1, rCode=0, trunc=0, maxSize=0)

    def test_bindAndTNamesStyle(self):
        m = self._makeMessage()
        m.queries = [dns.Query('fooby.com', dns.AXFR, dns.IN)]
        m.answers = self.records
        self.controller.messageReceived(m, None)
        self.assertEqual(self.results, self.records)

    def _gotResults(self, result):
        self.results = result

    def test_DJBStyle(self):
        records = self.records[:]
        while records:
            m = self._makeMessage()
            m.queries = []
            m.answers = [records.pop(0)]
            self.controller.messageReceived(m, None)
        self.assertEqual(self.results, self.records)