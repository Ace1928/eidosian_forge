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
def test_someRecordsWithTTLs(self):
    result_soa = copy.copy(my_soa)
    result_soa.ttl = my_soa.expire
    return self.namesTest(self.resolver.lookupAllRecords('my-domain.com'), [result_soa, dns.Record_A('1.2.3.4', ttl='1S'), dns.Record_NS('ns1.domain', ttl='2M'), dns.Record_NS('ns2.domain', ttl='3H'), dns.Record_SRV(257, 16383, 43690, 'some.other.place.fool', ttl='4D')])