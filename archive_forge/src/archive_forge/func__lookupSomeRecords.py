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
def _lookupSomeRecords(self, method, soa, makeRecord, target, addresses):
    """
        Perform a DNS lookup against a L{FileAuthority} configured with records
        as defined by C{makeRecord} and C{addresses}.

        @param method: The name of the lookup method to use; for example,
            C{"lookupNameservers"}.
        @type method: L{str}

        @param soa: A L{Record_SOA} for the zone for which the L{FileAuthority}
            is authoritative.

        @param makeRecord: A one-argument callable which accepts a name and
            returns an L{IRecord} provider.  L{FileAuthority} is constructed
            with this record.  The L{FileAuthority} is queried for a record of
            the resulting type with the given name.

        @param target: The extra name which the record returned by
            C{makeRecord} will be pointed at; this is the name which might
            require extra processing by the server so that all the available,
            useful information is returned.  For example, this is the target of
            a CNAME record or the mail exchange host pointed to by an MX record.
        @type target: L{bytes}

        @param addresses: A L{list} of records giving addresses of C{target}.

        @return: A L{Deferred} that fires with the result of the resolver
            method give by C{method}.
        """
    authority = NoFileAuthority(soa=(soa.mname.name, soa), records={soa.mname.name: [makeRecord(target)], target: addresses})
    return getattr(authority, method)(soa_record.mname.name)