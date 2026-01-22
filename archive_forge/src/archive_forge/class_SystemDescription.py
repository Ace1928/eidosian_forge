import struct
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@lldp.set_tlv_type(LLDP_TLV_SYSTEM_DESCRIPTION)
class SystemDescription(LLDPBasicTLV):
    """System description TLV encoder/decoder class

    =================== =====================================
    Attribute           Description
    =================== =====================================
    buf                 Binary data to parse.
    system_description  System description.
    =================== =====================================
    """
    _LEN_MAX = 255

    def __init__(self, buf=None, *args, **kwargs):
        super(SystemDescription, self).__init__(buf, *args, **kwargs)
        if buf:
            pass
        else:
            self.system_description = kwargs['system_description']
            self.len = len(self.system_description)
            assert self._len_valid()
            self.typelen = self.tlv_type << LLDP_TLV_TYPE_SHIFT | self.len

    def serialize(self):
        return struct.pack('!H', self.typelen) + self.tlv_info

    @property
    def system_description(self):
        return self.tlv_info

    @system_description.setter
    def system_description(self, value):
        self.tlv_info = value