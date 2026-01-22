import struct
from os_ken import utils
from os_ken.lib import type_desc
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import ofproto_common
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto.ofproto_parser import StringifyMixin
class _NXFlowSpec(StringifyMixin):
    _hdr_fmt_str = '!H'
    _dst_type = None
    _subclasses = {}
    _TYPE = {'nx-flow-spec-field': ['src', 'dst']}

    def __init__(self, src, dst, n_bits):
        self.src = src
        self.dst = dst
        self.n_bits = n_bits

    @classmethod
    def register(cls, subcls):
        assert issubclass(subcls, cls)
        assert subcls._dst_type not in cls._subclasses
        cls._subclasses[subcls._dst_type] = subcls

    @classmethod
    def parse(cls, buf):
        hdr, = struct.unpack_from(cls._hdr_fmt_str, buf, 0)
        rest = buf[struct.calcsize(cls._hdr_fmt_str):]
        if hdr == 0:
            return (None, rest)
        src_type = hdr >> 13 & 1
        dst_type = hdr >> 11 & 3
        n_bits = hdr & 1023
        subcls = cls._subclasses[dst_type]
        if src_type == 0:
            src = cls._parse_subfield(rest)
            rest = rest[6:]
        elif src_type == 1:
            src_len = (n_bits + 15) // 16 * 2
            src_bin = rest[:src_len]
            src = type_desc.IntDescr(size=src_len).to_user(src_bin)
            rest = rest[src_len:]
        if dst_type == 0:
            dst = cls._parse_subfield(rest)
            rest = rest[6:]
        elif dst_type == 1:
            dst = cls._parse_subfield(rest)
            rest = rest[6:]
        elif dst_type == 2:
            dst = ''
        return (subcls(src=src, dst=dst, n_bits=n_bits), rest)

    def serialize(self):
        buf = bytearray()
        if isinstance(self.src, tuple):
            src_type = 0
        else:
            src_type = 1
        val = src_type << 13 | self._dst_type << 11 | self.n_bits
        msg_pack_into(self._hdr_fmt_str, buf, 0, val)
        if src_type == 0:
            buf += self._serialize_subfield(self.src)
        elif src_type == 1:
            src_len = (self.n_bits + 15) // 16 * 2
            buf += type_desc.IntDescr(size=src_len).from_user(self.src)
        if self._dst_type == 0:
            buf += self._serialize_subfield(self.dst)
        elif self._dst_type == 1:
            buf += self._serialize_subfield(self.dst)
        elif self._dst_type == 2:
            pass
        return buf

    @staticmethod
    def _parse_subfield(buf):
        n, len = ofp.oxm_parse_header(buf, 0)
        assert len == 4
        field = ofp.oxm_to_user_header(n)
        rest = buf[len:]
        ofs, = struct.unpack_from('!H', rest, 0)
        return (field, ofs)

    @staticmethod
    def _serialize_subfield(subfield):
        field, ofs = subfield
        buf = bytearray()
        n = ofp.oxm_from_user_header(field)
        ofp.oxm_serialize_header(n, buf, 0)
        assert len(buf) == 4
        msg_pack_into('!H', buf, 4, ofs)
        return buf