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
class Record_SOA(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    Marks the start of a zone of authority.

    This record describes parameters which are shared by all records within a
    particular zone.

    @type mname: L{Name}
    @ivar mname: The domain-name of the name server that was the original or
        primary source of data for this zone.

    @type rname: L{Name}
    @ivar rname: A domain-name which specifies the mailbox of the person
        responsible for this zone.

    @type serial: L{int}
    @ivar serial: The unsigned 32 bit version number of the original copy of
        the zone.  Zone transfers preserve this value.  This value wraps and
        should be compared using sequence space arithmetic.

    @type refresh: L{int}
    @ivar refresh: A 32 bit time interval before the zone should be refreshed.

    @type minimum: L{int}
    @ivar minimum: The unsigned 32 bit minimum TTL field that should be
        exported with any RR from this zone.

    @type expire: L{int}
    @ivar expire: A 32 bit time value that specifies the upper limit on the
        time interval that can elapse before the zone is no longer
        authoritative.

    @type retry: L{int}
    @ivar retry: A 32 bit time interval that should elapse before a failed
        refresh should be retried.

    @type ttl: L{int}
    @ivar ttl: The default TTL to use for records served from this zone.
    """
    fancybasename = 'SOA'
    compareAttributes = ('serial', 'mname', 'rname', 'refresh', 'expire', 'retry', 'minimum', 'ttl')
    showAttributes = (('mname', 'mname', '%s'), ('rname', 'rname', '%s'), 'serial', 'refresh', 'retry', 'expire', 'minimum', 'ttl')
    TYPE = SOA

    def __init__(self, mname=b'', rname=b'', serial=0, refresh=0, retry=0, expire=0, minimum=0, ttl=None):
        """
        @param mname: See L{Record_SOA.mname}
        @type mname: L{bytes} or L{str}

        @param rname: See L{Record_SOA.rname}
        @type rname: L{bytes} or L{str}
        """
        self.mname, self.rname = (Name(mname), Name(rname))
        self.serial, self.refresh = (str2time(serial), str2time(refresh))
        self.minimum, self.expire = (str2time(minimum), str2time(expire))
        self.retry = str2time(retry)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        self.mname.encode(strio, compDict)
        self.rname.encode(strio, compDict)
        strio.write(struct.pack('!LlllL', self.serial, self.refresh, self.retry, self.expire, self.minimum))

    def decode(self, strio, length=None):
        self.mname, self.rname = (Name(), Name())
        self.mname.decode(strio)
        self.rname.decode(strio)
        r = struct.unpack('!LlllL', readPrecisely(strio, 20))
        self.serial, self.refresh, self.retry, self.expire, self.minimum = r

    def __hash__(self):
        return hash((self.serial, self.mname, self.rname, self.refresh, self.expire, self.retry))