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
def test_recordMissing(self):
    """
        If a L{FileAuthority} has a zone which includes an I{NS} record for a
        particular name and that authority is asked for another record for the
        same name which does not exist, the I{NS} record is not included in the
        authority section of the response.
        """
    authority = NoFileAuthority(soa=(soa_record.mname.name, soa_record), records={soa_record.mname.name: [soa_record, dns.Record_NS('1.2.3.4')]})
    answer, authority, additional = self.successResultOf(authority.lookupAddress(soa_record.mname.name))
    self.assertEqual(answer, [])
    self.assertEqual(authority, [dns.RRHeader(soa_record.mname.name, soa_record.TYPE, ttl=soa_record.expire, payload=soa_record, auth=True)])
    self.assertEqual(additional, [])