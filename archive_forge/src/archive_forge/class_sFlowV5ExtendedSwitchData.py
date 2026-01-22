import struct
import logging
class sFlowV5ExtendedSwitchData(object):
    _PACK_STR = '!IIII'

    def __init__(self, src_vlan, src_priority, dest_vlan, dest_priority):
        super(sFlowV5ExtendedSwitchData, self).__init__()
        self.src_vlan = src_vlan
        self.src_priority = src_priority
        self.dest_vlan = dest_vlan
        self.dest_priority = dest_priority

    @classmethod
    def parser(cls, buf, offset):
        src_vlan, src_priority, dest_vlan, dest_priority = struct.unpack_from(cls._PACK_STR, buf, offset)
        msg = cls(src_vlan, src_priority, dest_vlan, dest_priority)
        return msg