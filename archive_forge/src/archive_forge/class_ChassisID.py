import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_CHASSIS_ID)
class ChassisID(LLDPBasicTLV):
    """Chassis ID TLV encoder/decoder class

    ============== =====================================
    Attribute      Description
    ============== =====================================
    buf            Binary data to parse.
    subtype        Subtype.
    chassis_id     Chassis id corresponding to subtype.
    ============== =====================================
    """
    _PACK_STR = '!B'
    _PACK_SIZE = struct.calcsize(_PACK_STR)
    _LEN_MIN = 2
    _LEN_MAX = 256
    SUB_CHASSIS_COMPONENT = 1
    SUB_INTERFACE_ALIAS = 2
    SUB_PORT_COMPONENT = 3
    SUB_MAC_ADDRESS = 4
    SUB_NETWORK_ADDRESS = 5
    SUB_INTERFACE_NAME = 6
    SUB_LOCALLY_ASSIGNED = 7

    def __init__(self, buf=None, *args, **kwargs):
        super(ChassisID, self).__init__(buf, *args, **kwargs)
        if buf:
            self.subtype, = struct.unpack(self._PACK_STR, self.tlv_info[:self._PACK_SIZE])
            self.chassis_id = self.tlv_info[self._PACK_SIZE:]
        else:
            self.subtype = kwargs['subtype']
            self.chassis_id = kwargs['chassis_id']
            self.len = self._PACK_SIZE + len(self.chassis_id)
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!HB', self.typelen, self.subtype) + self.chassis_id