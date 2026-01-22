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
class Record_WKS(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    A well known service description.

    This record type is obsolete.  See L{Record_SRV}.

    @type address: L{bytes}
    @ivar address: The packed network-order representation of the IPv4 address
        associated with this record.

    @type protocol: L{int}
    @ivar protocol: The 8 bit IP protocol number for which this service map is
        relevant.

    @type map: L{bytes}
    @ivar map: A bitvector indicating the services available at the specified
        address.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    fancybasename = 'WKS'
    compareAttributes = ('address', 'protocol', 'map', 'ttl')
    showAttributes = [('_address', 'address', '%s'), 'protocol', 'ttl']
    TYPE = WKS

    @property
    def _address(self):
        return socket.inet_ntoa(self.address)

    def __init__(self, address='0.0.0.0', protocol=0, map=b'', ttl=None):
        """
        @type address: L{bytes} or L{str}
        @param address: The IPv4 address associated with this record, in
            quad-dotted notation.
        """
        if isinstance(address, bytes):
            address = address.decode('idna')
        self.address = socket.inet_aton(address)
        self.protocol, self.map = (protocol, map)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(self.address)
        strio.write(struct.pack('!B', self.protocol))
        strio.write(self.map)

    def decode(self, strio, length=None):
        self.address = readPrecisely(strio, 4)
        self.protocol = struct.unpack('!B', readPrecisely(strio, 1))[0]
        self.map = readPrecisely(strio, length - 5)

    def __hash__(self):
        return hash((self.address, self.protocol, self.map))