import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class operation(stringify.StringifyMixin, metaclass=abc.ABCMeta):
    _TLV_TYPES = {}
    _END_TLV_LEN = 1

    @staticmethod
    def register_tlv_types(type_):

        def _register_tlv_types(cls):
            operation._TLV_TYPES[type_] = cls
            return cls
        return _register_tlv_types

    def __init__(self, md_lv, version, tlvs):
        self.md_lv = md_lv
        self.version = version
        tlvs = tlvs or []
        assert isinstance(tlvs, list)
        for tlv_ in tlvs:
            assert isinstance(tlv_, tlv)
        self.tlvs = tlvs

    @classmethod
    @abc.abstractmethod
    def parser(cls, buf):
        pass

    @abc.abstractmethod
    def serialize(self):
        pass

    @abc.abstractmethod
    def __len__(self):
        pass

    @classmethod
    def _parser_tlvs(cls, buf):
        offset = 0
        tlvs = []
        while True:
            type_, = struct.unpack_from('!B', buf, offset)
            cls_ = cls._TLV_TYPES.get(type_)
            if not cls_:
                assert type_ is CFM_END_TLV
                break
            tlv_ = cls_.parser(buf[offset:])
            tlvs.append(tlv_)
            offset += len(tlv_)
        return tlvs

    @staticmethod
    def _serialize_tlvs(tlvs):
        buf = bytearray()
        for tlv_ in tlvs:
            buf.extend(tlv_.serialize())
        return buf

    def _calc_len(self, len_):
        for tlv_ in self.tlvs:
            len_ += len(tlv_)
        len_ += self._END_TLV_LEN
        return len_