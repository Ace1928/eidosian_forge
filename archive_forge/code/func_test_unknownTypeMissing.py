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
def test_unknownTypeMissing(self):
    """
        Requesting a record of unknown type where other records exist for the
        name in question results in an empty answer set.
        """
    unknownType = max(common.typeToMethod) + 1
    answer, authority, additional = self.successResultOf(my_domain_com.query(Query(name='my-domain.com', type=unknownType)))
    self.assertEqual(answer, [])