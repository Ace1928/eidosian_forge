import struct
import base64
from os_ken.lib import addrconv
from os_ken.lib import mac
from os_ken.lib.pack_utils import msg_pack_into
from os_ken.lib.packet import packet
from os_ken import exception
from os_ken import utils
from os_ken.ofproto.ofproto_parser import StringifyMixin, MsgBase
from os_ken.ofproto import ether
from os_ken.ofproto import nicira_ext
from os_ken.ofproto import nx_actions
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_common
from os_ken.ofproto import ofproto_v1_3 as ofproto
import logging
class OFPMatchField(StringifyMixin):
    _FIELDS_HEADERS = {}

    @staticmethod
    def register_field_header(headers):

        def _register_field_header(cls):
            for header in headers:
                OFPMatchField._FIELDS_HEADERS[header] = cls
            return cls
        return _register_field_header

    def __init__(self, header):
        self.header = header
        self.n_bytes = ofproto.oxm_tlv_header_extract_length(header)
        self.length = 0

    @classmethod
    def cls_to_header(cls, cls_, hasmask):
        inv = dict(((v, k) for k, v in cls._FIELDS_HEADERS.items() if (k >> 8 & 1 != 0) == hasmask))
        return inv[cls_]

    @staticmethod
    def make(header, value, mask=None):
        cls_ = OFPMatchField._FIELDS_HEADERS.get(header)
        return cls_(header, value, mask)

    @classmethod
    def parser(cls, buf, offset):
        header, = struct.unpack_from('!I', buf, offset)
        cls_ = OFPMatchField._FIELDS_HEADERS.get(header)
        if cls_:
            field = cls_.field_parser(header, buf, offset)
        else:
            field = OFPMatchField(header)
        field.length = (header & 255) + 4
        return field

    @classmethod
    def field_parser(cls, header, buf, offset):
        mask = None
        if ofproto.oxm_tlv_header_extract_hasmask(header):
            pack_str = '!' + cls.pack_str[1:] * 2
            value, mask = struct.unpack_from(pack_str, buf, offset + 4)
        else:
            value, = struct.unpack_from(cls.pack_str, buf, offset + 4)
        return cls(header, value, mask)

    def serialize(self, buf, offset):
        if ofproto.oxm_tlv_header_extract_hasmask(self.header):
            self.put_w(buf, offset, self.value, self.mask)
        else:
            self.put(buf, offset, self.value)

    def _put_header(self, buf, offset):
        msg_pack_into('!I', buf, offset, self.header)
        self.length = 4

    def _put(self, buf, offset, value):
        msg_pack_into(self.pack_str, buf, offset, value)
        self.length += self.n_bytes

    def put_w(self, buf, offset, value, mask):
        self._put_header(buf, offset)
        self._put(buf, offset + self.length, value)
        self._put(buf, offset + self.length, mask)

    def put(self, buf, offset, value):
        self._put_header(buf, offset)
        self._put(buf, offset + self.length, value)

    def _putv6(self, buf, offset, value):
        msg_pack_into(self.pack_str, buf, offset, *value)
        self.length += self.n_bytes

    def putv6(self, buf, offset, value, mask=None):
        self._put_header(buf, offset)
        self._putv6(buf, offset + self.length, value)
        if mask and len(mask):
            self._putv6(buf, offset + self.length, mask)

    def oxm_len(self):
        return self.header & 255

    def to_jsondict(self):
        d = super(OFPMatchField, self).to_jsondict()
        v = d[self.__class__.__name__]
        del v['header']
        del v['length']
        del v['n_bytes']
        return d

    @classmethod
    def from_jsondict(cls, dict_):
        return {cls.__name__: dict_}

    def stringify_attrs(self):
        f = super(OFPMatchField, self).stringify_attrs
        if not ofproto.oxm_tlv_header_extract_hasmask(self.header):

            def g():
                for k, v in f():
                    if k != 'mask':
                        yield (k, v)
            return g()
        else:
            return f()