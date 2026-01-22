import base64
import enum
import struct
import dns.enum
import dns.exception
import dns.immutable
import dns.ipv4
import dns.ipv6
import dns.name
import dns.rdata
import dns.rdtypes.util
import dns.renderer
import dns.tokenizer
import dns.wire
@dns.immutable.immutable
class GenericParam(Param):
    """Generic SVCB parameter"""

    def __init__(self, value):
        self.value = dns.rdata.Rdata._as_bytes(value, True)

    @classmethod
    def emptiness(cls):
        return Emptiness.ALLOWED

    @classmethod
    def from_value(cls, value):
        if value is None or len(value) == 0:
            return None
        else:
            return cls(_unescape(value))

    def to_text(self):
        return '"' + dns.rdata._escapify(self.value) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        value = parser.get_bytes(parser.remaining())
        if len(value) == 0:
            return None
        else:
            return cls(value)

    def to_wire(self, file, origin=None):
        file.write(self.value)