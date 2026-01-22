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
@implementer(IEncodable)
class RRHeader(tputil.FancyEqMixin):
    """
    A resource record header.

    @cvar fmt: L{str} specifying the byte format of an RR.

    @ivar name: The name about which this reply contains information.
    @type name: L{Name}

    @ivar type: The query type of the original request.
    @type type: L{int}

    @ivar cls: The query class of the original request.

    @ivar ttl: The time-to-live for this record.
    @type ttl: L{int}

    @ivar payload: The record described by this header.
    @type payload: L{IEncodableRecord} or L{None}

    @ivar auth: A L{bool} indicating whether this C{RRHeader} was parsed from
        an authoritative message.
    """
    compareAttributes = ('name', 'type', 'cls', 'ttl', 'payload', 'auth')
    fmt = '!HHIH'
    rdlength = None
    cachedResponse = None

    def __init__(self, name: Union[bytes, str]=b'', type: int=A, cls: int=IN, ttl: SupportsInt=0, payload: Optional[IEncodableRecord]=None, auth: bool=False):
        """
        @type name: L{bytes} or L{str}
        @param name: See L{RRHeader.name}

        @type type: L{int}
        @param type: The query type.

        @type cls: L{int}
        @param cls: The query class.

        @type ttl: L{int}
        @param ttl: Time to live for this record.  This will be
            converted to an L{int}.

        @type payload: L{IEncodableRecord} or L{None}
        @param payload: An optional Query Type specific data object.

        @raises TypeError: if the ttl cannot be converted to an L{int}.
        @raises ValueError: if the ttl is negative.
        @raises ValueError: if the payload type is not equal to the C{type}
                            argument.
        """
        payloadType = None if payload is None else payload.TYPE
        if payloadType is not None and payloadType != type:
            raise ValueError('Payload type (%s) does not match given type (%s)' % (QUERY_TYPES.get(payloadType, payloadType), QUERY_TYPES.get(type, type)))
        integralTTL = int(ttl)
        if integralTTL < 0:
            raise ValueError('TTL cannot be negative')
        self.name = Name(name)
        self.type = type
        self.cls = cls
        self.ttl = integralTTL
        self.payload = payload
        self.auth = auth

    def encode(self, strio, compDict=None):
        self.name.encode(strio, compDict)
        strio.write(struct.pack(self.fmt, self.type, self.cls, self.ttl, 0))
        if self.payload:
            prefix = strio.tell()
            self.payload.encode(strio, compDict)
            aft = strio.tell()
            strio.seek(prefix - 2, 0)
            strio.write(struct.pack('!H', aft - prefix))
            strio.seek(aft, 0)

    def decode(self, strio, length=None):
        self.name.decode(strio)
        l = struct.calcsize(self.fmt)
        buff = readPrecisely(strio, l)
        r = struct.unpack(self.fmt, buff)
        self.type, self.cls, self.ttl, self.rdlength = r

    def isAuthoritative(self):
        return self.auth

    def __str__(self) -> str:
        t = QUERY_TYPES.get(self.type, EXT_QUERIES.get(self.type, 'UNKNOWN (%d)' % self.type))
        c = QUERY_CLASSES.get(self.cls, 'UNKNOWN (%d)' % self.cls)
        return '<RR name=%s type=%s class=%s ttl=%ds auth=%s>' % (self.name, t, c, self.ttl, self.auth and 'True' or 'False')
    __repr__ = __str__