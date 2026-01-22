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
class Record_MX(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    Mail exchange.

    @type preference: L{int}
    @ivar preference: Specifies the preference given to this RR among others at
        the same owner.  Lower values are preferred.

    @type name: L{Name}
    @ivar name: A domain-name which specifies a host willing to act as a mail
        exchange.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.
    """
    TYPE = MX
    fancybasename = 'MX'
    compareAttributes = ('preference', 'name', 'ttl')
    showAttributes = ('preference', ('name', 'name', '%s'), 'ttl')

    def __init__(self, preference=0, name=b'', ttl=None, **kwargs):
        """
        @param name: See L{Record_MX.name}.
        @type name: L{bytes} or L{str}
        """
        self.preference = int(preference)
        self.name = Name(kwargs.get('exchange', name))
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!H', self.preference))
        self.name.encode(strio, compDict)

    def decode(self, strio, length=None):
        self.preference = struct.unpack('!H', readPrecisely(strio, 2))[0]
        self.name = Name()
        self.name.decode(strio)

    def __hash__(self):
        return hash((self.preference, self.name))