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
class MandatoryParam(Param):

    def __init__(self, keys):
        keys = sorted([_validate_key(key)[0] for key in keys])
        prior_k = None
        for k in keys:
            if k == prior_k:
                raise ValueError(f'duplicate key {k:d}')
            prior_k = k
            if k == ParamKey.MANDATORY:
                raise ValueError('listed the mandatory key as mandatory')
        self.keys = tuple(keys)

    @classmethod
    def from_value(cls, value):
        keys = [k.encode() for k in value.split(',')]
        return cls(keys)

    def to_text(self):
        return '"' + ','.join([key_to_text(key) for key in self.keys]) + '"'

    @classmethod
    def from_wire_parser(cls, parser, origin=None):
        keys = []
        last_key = -1
        while parser.remaining() > 0:
            key = parser.get_uint16()
            if key < last_key:
                raise dns.exception.FormError('manadatory keys not ascending')
            last_key = key
            keys.append(key)
        return cls(keys)

    def to_wire(self, file, origin=None):
        for key in self.keys:
            file.write(struct.pack('!H', key))