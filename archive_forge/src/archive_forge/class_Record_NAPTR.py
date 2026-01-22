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
class Record_NAPTR(tputil.FancyEqMixin, tputil.FancyStrMixin):
    """
    The location of the server(s) for a specific protocol and domain.

    @type order: L{int}
    @ivar order: An integer specifying the order in which the NAPTR records
        MUST be processed to ensure the correct ordering of rules.  Low numbers
        are processed before high numbers.

    @type preference: L{int}
    @ivar preference: An integer that specifies the order in which NAPTR
        records with equal "order" values SHOULD be processed, low numbers
        being processed before high numbers.

    @type flag: L{Charstr}
    @ivar flag: A <character-string> containing flags to control aspects of the
        rewriting and interpretation of the fields in the record.  Flags
        are single characters from the set [A-Z0-9].  The case of the alphabetic
        characters is not significant.

        At this time only four flags, "S", "A", "U", and "P", are defined.

    @type service: L{Charstr}
    @ivar service: Specifies the service(s) available down this rewrite path.
        It may also specify the particular protocol that is used to talk with a
        service.  A protocol MUST be specified if the flags field states that
        the NAPTR is terminal.

    @type regexp: L{Charstr}
    @ivar regexp: A STRING containing a substitution expression that is applied
        to the original string held by the client in order to construct the
        next domain name to lookup.

    @type replacement: L{Name}
    @ivar replacement: The next NAME to query for NAPTR, SRV, or address
        records depending on the value of the flags field.  This MUST be a
        fully qualified domain-name.

    @type ttl: L{int}
    @ivar ttl: The maximum number of seconds which this record should be
        cached.

    @see: U{http://www.faqs.org/rfcs/rfc2915.html}
    """
    TYPE = NAPTR
    compareAttributes = ('order', 'preference', 'flags', 'service', 'regexp', 'replacement')
    fancybasename = 'NAPTR'
    showAttributes = ('order', 'preference', ('flags', 'flags', '%s'), ('service', 'service', '%s'), ('regexp', 'regexp', '%s'), ('replacement', 'replacement', '%s'), 'ttl')

    def __init__(self, order=0, preference=0, flags=b'', service=b'', regexp=b'', replacement=b'', ttl=None):
        """
        @param replacement: See L{Record_NAPTR.replacement}
        @type replacement: L{bytes} or L{str}
        """
        self.order = int(order)
        self.preference = int(preference)
        self.flags = Charstr(flags)
        self.service = Charstr(service)
        self.regexp = Charstr(regexp)
        self.replacement = Name(replacement)
        self.ttl = str2time(ttl)

    def encode(self, strio, compDict=None):
        strio.write(struct.pack('!HH', self.order, self.preference))
        self.flags.encode(strio, None)
        self.service.encode(strio, None)
        self.regexp.encode(strio, None)
        self.replacement.encode(strio, None)

    def decode(self, strio, length=None):
        r = struct.unpack('!HH', readPrecisely(strio, struct.calcsize('!HH')))
        self.order, self.preference = r
        self.flags = Charstr()
        self.service = Charstr()
        self.regexp = Charstr()
        self.replacement = Name()
        self.flags.decode(strio)
        self.service.decode(strio)
        self.regexp.decode(strio)
        self.replacement.decode(strio)

    def __hash__(self):
        return hash((self.order, self.preference, self.flags, self.service, self.regexp, self.replacement))