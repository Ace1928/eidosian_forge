from __future__ import annotations
import inspect
import random
import socket
import struct
from io import BytesIO
from itertools import chain
from typing import Optional, Sequence, SupportsInt, Union, overload
from zope.interface import Attribute, Interface, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import CannotListenError
from twisted.python import failure, log, randbytes, util as tputil
from twisted.python.compat import cmp, comparable, nativeString
from twisted.names.error import (
@implementer(IEncodableRecord)
class Record_AFSDB(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    Map from a domain name to the name of an AFS cell database server.

    @type subtype: L{int}
    @ivar subtype: In the case of subtype 1, the host has an AFS version 3.0
        Volume Location Server for the named AFS cell.  In the case of subtype
        2, the host has an authenticated name server holding the cell-root
        directory node for the named DCE/NCA cell.

    @type hostname: L{Name}
    @ivar hostname: The domain name of a host that has a server for the cell
        named by this record.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc1183.html}
    """
    TYPE = AFSDB
    fancybasename = 'AFSDB'
    compareAttributes = ('subtype', 'hostname', 'ttl')
    showAttributes = ('subtype', ('hostname', 'hostname', '%s'), 'ttl')

    def __init__(self, subtype=0, hostname=b'', ttl=None):
        """
        @param hostname: See L{Record_AFSDB.hostname}
        @type hostname: L{bytes} or L{str}
        """
        self.subtype = int(subtype)
        self.hostname = Name(hostname)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!H', self.subtype))
        self.hostname.encode(strio, compDict)

    def decode(self, strio, length=None):
        r = struct.unpack('!H', readPrecisely(strio, struct.calcsize('!H')))
        self.subtype, = r
        self.hostname.decode(strio)

    def __hash__(self):
        return hash((self.subtype, self.hostname))