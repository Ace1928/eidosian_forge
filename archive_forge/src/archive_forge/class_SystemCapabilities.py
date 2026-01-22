import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_SYSTEM_CAPABILITIES)
class SystemCapabilities(LLDPBasicTLV):
    """System Capabilities TLV encoder/decoder class

    ================= =====================================
    Attribute         Description
    ================= =====================================
    buf               Binary data to parse.
    system_cap        System Capabilities.
    enabled_cap       Enabled Capabilities.
    ================= =====================================
    """
    _PACK_STR = '!HH'
    _PACK_SIZE = struct.calcsize(_PACK_STR)
    _LEN_MIN = _PACK_SIZE
    _LEN_MAX = _PACK_SIZE
    CAP_REPEATER = 1 << 1
    CAP_MAC_BRIDGE = 1 << 2
    CAP_WLAN_ACCESS_POINT = 1 << 3
    CAP_ROUTER = 1 << 4
    CAP_TELEPHONE = 1 << 5
    CAP_DOCSIS = 1 << 6
    CAP_STATION_ONLY = 1 << 7
    CAP_CVLAN = 1 << 8
    CAP_SVLAN = 1 << 9
    CAP_TPMR = 1 << 10

    def __init__(self, buf=None, *args, **kwargs):
        super(SystemCapabilities, self).__init__(buf, *args, **kwargs)
        if buf:
            self.system_cap, self.enabled_cap = struct.unpack(self._PACK_STR, self.tlv_info[:self._PACK_SIZE])
        else:
            self.system_cap = kwargs['system_cap']
            self.enabled_cap = kwargs['enabled_cap']
            self.len = self._PACK_SIZE
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!HHH', self.typelen, self.system_cap, self.enabled_cap)