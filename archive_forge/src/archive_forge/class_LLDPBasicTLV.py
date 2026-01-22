import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
class LLDPBasicTLV(stringify.StringifyMixin):
    _LEN_MIN = 0
    _LEN_MAX = 511
    tlv_type = None

    def __init__(self, buf=None, *_args, **_kwargs):
        super(LLDPBasicTLV, self).__init__()
        if buf:
            self.typelen, = struct.unpack(LLDP_TLV_TYPELEN_STR, buf[:LLDP_TLV_SIZE])
            tlv_type = (self.typelen & LLDP_TLV_TYPE_MASK) >> LLDP_TLV_TYPE_SHIFT
            assert self.tlv_type == tlv_type
            self.len = self.typelen & LLDP_TLV_LENGTH_MASK
            assert len(buf) >= self.len + LLDP_TLV_SIZE
            self.tlv_info = buf[LLDP_TLV_SIZE:]
            self.tlv_info = self.tlv_info[:self.len]

    @staticmethod
    def get_type(buf):
        typelen, = struct.unpack(LLDP_TLV_TYPELEN_STR, buf[:LLDP_TLV_SIZE])
        return (typelen & LLDP_TLV_TYPE_MASK) >> LLDP_TLV_TYPE_SHIFT

    @staticmethod
    def set_tlv_type(subcls, tlv_type):
        assert issubclass(subcls, LLDPBasicTLV)
        subcls.tlv_type = tlv_type

    def _len_valid(self):
        return self._LEN_MIN <= self.len and self.len <= self._LEN_MAX