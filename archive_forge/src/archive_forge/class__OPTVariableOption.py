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
class _OPTVariableOption(tputil.FancyStrMixin, tputil.FancyEqMixin):
    """
    A class to represent OPT record variable options.

    @see: L{_OPTVariableOption.__init__} for documentation of public
        instance attributes.

    @see: U{https://tools.ietf.org/html/rfc6891#section-6.1.2}

    @since: 13.2
    """
    showAttributes = ('code', ('data', nativeString))
    compareAttributes = ('code', 'data')
    _fmt = '!HH'

    def __init__(self, code=0, data=b''):
        """
        @type code: L{int}
        @param code: The option code

        @type data: L{bytes}
        @param data: The option data
        """
        self.code = code
        self.data = data

    def encode(self, strio, compDict=None):
        """
        Encode this L{_OPTVariableOption} to bytes.

        @type strio: file
        @param strio: the byte representation of this
            L{_OPTVariableOption} will be written to this file.

        @type compDict: L{dict} or L{None}
        @param compDict: A dictionary of backreference addresses that
            have already been written to this stream and that may
            be used for DNS name compression.
        """
        strio.write(struct.pack(self._fmt, self.code, len(self.data)) + self.data)

    def decode(self, strio, length=None):
        """
        Decode bytes into an L{_OPTVariableOption} instance.

        @type strio: file
        @param strio: Bytes will be read from this file until the full
            L{_OPTVariableOption} is decoded.

        @type length: L{int} or L{None}
        @param length: Not used.
        """
        l = struct.calcsize(self._fmt)
        buff = readPrecisely(strio, l)
        self.code, length = struct.unpack(self._fmt, buff)
        self.data = readPrecisely(strio, length)