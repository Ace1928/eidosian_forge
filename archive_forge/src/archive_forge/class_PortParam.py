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
class PortParam(Param):

    def __init__(self, port):
        self.port = dns.rdata.Rdata._as_uint16(port)

    @classmethod
    def from_value(cls, value):
        value = int(value)
        return cls(value)

    def to_text(self):
        return f'"{self.port}"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        port = parser.get_uint16()
        return cls(port)

    def to_wire(self, file, origin=None):
        file.write(struct.pack('!H', self.port))