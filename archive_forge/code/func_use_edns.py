from __future__ import absolute_import
from io import StringIO
import struct
import time
import dns.edns
import dns.exception
import dns.flags
import dns.name
import dns.opcode
import dns.entropy
import dns.rcode
import dns.rdata
import dns.rdataclass
import dns.rdatatype
import dns.rrset
import dns.renderer
import dns.tsig
import dns.wiredata
from ._compat import long, xrange, string_types
def use_edns(self, edns=0, ednsflags=0, payload=1280, request_payload=None, options=None):
    """Configure EDNS behavior.

        *edns*, an ``int``, is the EDNS level to use.  Specifying
        ``None``, ``False``, or ``-1`` means "do not use EDNS", and in this case
        the other parameters are ignored.  Specifying ``True`` is
        equivalent to specifying 0, i.e. "use EDNS0".

        *ednsflags*, an ``int``, the EDNS flag values.

        *payload*, an ``int``, is the EDNS sender's payload field, which is the
        maximum size of UDP datagram the sender can handle.  I.e. how big
        a response to this message can be.

        *request_payload*, an ``int``, is the EDNS payload size to use when
        sending this message.  If not specified, defaults to the value of
        *payload*.

        *options*, a list of ``dns.edns.Option`` objects or ``None``, the EDNS
        options.
        """
    if edns is None or edns is False:
        edns = -1
    if edns is True:
        edns = 0
    if request_payload is None:
        request_payload = payload
    if edns < 0:
        ednsflags = 0
        payload = 0
        request_payload = 0
        options = []
    else:
        ednsflags &= long(4278255615)
        ednsflags |= edns << 16
        if options is None:
            options = []
    self.edns = edns
    self.ednsflags = ednsflags
    self.payload = payload
    self.options = options
    self.request_payload = request_payload