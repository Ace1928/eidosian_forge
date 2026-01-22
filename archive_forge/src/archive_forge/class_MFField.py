import struct
from os_ken import exception
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import inet
import logging
class MFField(object):
    _FIELDS_HEADERS = {}

    @staticmethod
    def register_field_header(headers):

        def _register_field_header(cls):
            for header in headers:
                MFField._FIELDS_HEADERS[header] = cls
            return cls
        return _register_field_header

    def __init__(self, nxm_header, pack_str):
        self.nxm_header = nxm_header
        self.pack_str = pack_str
        self.n_bytes = struct.calcsize(pack_str)
        self.n_bits = self.n_bytes * 8

    @classmethod
    def parser(cls, buf, offset):
        header, = struct.unpack_from('!I', buf, offset)
        cls_ = MFField._FIELDS_HEADERS.get(header)
        if cls_:
            field = cls_.field_parser(header, buf, offset)
        else:
            raise
        field.length = (header & 255) + 4
        return field

    @classmethod
    def field_parser(cls, header, buf, offset):
        hasmask = header >> 8 & 1
        mask = None
        if hasmask:
            pack_str = '!' + cls.pack_str[1:] * 2
            value, mask = struct.unpack_from(pack_str, buf, offset + 4)
        else:
            value, = struct.unpack_from(cls.pack_str, buf, offset + 4)
        return cls(header, value, mask)

    def _put(self, buf, offset, value):
        msg_pack_into(self.pack_str, buf, offset, value)
        return self.n_bytes

    def putw(self, buf, offset, value, mask):
        len_ = self._put(buf, offset, value)
        return len_ + self._put(buf, offset + len_, mask)

    def _is_all_ones(self, value):
        return value == (1 << self.n_bits) - 1

    def putm(self, buf, offset, value, mask):
        if mask == 0:
            return 0
        elif self._is_all_ones(mask):
            return self._put(buf, offset, value)
        else:
            return self.putw(buf, offset, value, mask)

    def _putv6(self, buf, offset, value):
        msg_pack_into(self.pack_str, buf, offset, *value)
        return self.n_bytes

    def putv6(self, buf, offset, value, mask):
        len_ = self._putv6(buf, offset, value)
        if len(mask):
            return len_ + self._putv6(buf, offset + len_, mask)
        return len_