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
class Record_A6(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    An IPv6 address.

    This is an experimental record type.

    @type prefixLen: L{int}
    @ivar prefixLen: The length of the suffix.

    @type suffix: L{bytes}
    @ivar suffix: An IPv6 address suffix in network order.

    @type prefix: L{Name}
    @ivar prefix: If specified, a name which will be used as a prefix for other
        A6 records.

    @type bytes: L{int}
    @ivar bytes: The length of the prefix.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc2874.html}
    @see: U{http://www.faqs.org/rfcs/rfc3363.html}
    @see: U{http://www.faqs.org/rfcs/rfc3364.html}
    """
    TYPE = A6
    fancybasename = 'A6'
    showAttributes = (('_suffix', 'suffix', '%s'), ('prefix', 'prefix', '%s'), 'ttl')
    compareAttributes = ('prefixLen', 'prefix', 'suffix', 'ttl')

    @property
    def _suffix(self):
        return socket.inet_ntop(AF_INET6, self.suffix)

    def __init__(self, prefixLen: int=0, suffix: bytes | str='::', prefix: bytes | str=b'', ttl: Union[str, bytes, int, None]=None):
        """
        @param suffix: An IPv6 address suffix in in RFC 2373 format.
        @type suffix: L{bytes} or L{str}

        @param prefix: An IPv6 address prefix for other A6 records.
        @type prefix: L{bytes} or L{str}
        """
        if isinstance(suffix, bytes):
            suffix = suffix.decode('idna')
        self.prefixLen = prefixLen
        self.suffix = socket.inet_pton(AF_INET6, suffix)
        self.prefix = Name(prefix)
        self.bytes = int((128 - self.prefixLen) / 8.0)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!B', self.prefixLen))
        if self.bytes:
            strio.write(self.suffix[-self.bytes:])
        if self.prefixLen:
            self.prefix.encode(strio, None)

    def decode(self, strio, length=None):
        self.prefixLen = struct.unpack('!B', readPrecisely(strio, 1))[0]
        self.bytes = int((128 - self.prefixLen) / 8.0)
        if self.bytes:
            self.suffix = b'\x00' * (16 - self.bytes) + readPrecisely(strio, self.bytes)
        if self.prefixLen:
            self.prefix.decode(strio)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Record_A6):
            return self.prefixLen == other.prefixLen and self.suffix[-self.bytes:] == other.suffix[-self.bytes:] and (self.prefix == other.prefix) and (self.ttl == other.ttl)
        return NotImplemented

    def __hash__(self):
        return hash((self.prefixLen, self.suffix[-self.bytes:], self.prefix))

    def __str__(self) -> str:
        return '<A6 %s %s (%d) ttl=%s>' % (self.prefix, socket.inet_ntop(AF_INET6, self.suffix), self.prefixLen, self.ttl)