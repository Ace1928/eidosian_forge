import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_TTL)
class TTL(LLDPBasicTLV):
    """Time To Live TLV encoder/decoder class

    ============== =====================================
    Attribute      Description
    ============== =====================================
    buf            Binary data to parse.
    ttl            Time To Live.
    ============== =====================================
    """
    _PACK_STR = '!H'
    _PACK_SIZE = struct.calcsize(_PACK_STR)
    _LEN_MIN = _PACK_SIZE
    _LEN_MAX = _PACK_SIZE

    def __init__(self, buf=None, *args, **kwargs):
        super(TTL, self).__init__(buf, *args, **kwargs)
        if buf:
            self.ttl, = struct.unpack(self._PACK_STR, self.tlv_info[:self._PACK_SIZE])
        else:
            self.ttl = kwargs['ttl']
            self.len = self._PACK_SIZE
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!HH', self.typelen, self.ttl)