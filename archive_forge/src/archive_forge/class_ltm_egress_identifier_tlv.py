import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_LTM_EGRESS_IDENTIFIER_TLV)
class ltm_egress_identifier_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) LTM EGRESS TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    length         Length of Value field.
                   (0 means automatically-calculate when encoding.)
    egress_id_ui   Egress Identifier of Unique ID.
    egress_id_mac  Egress Identifier of MAC address.
    ============== =======================================
    """
    _PACK_STR = '!BHH6s'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, length=0, egress_id_ui=0, egress_id_mac='00:00:00:00:00:00'):
        super(ltm_egress_identifier_tlv, self).__init__(length)
        self._type = CFM_LTM_EGRESS_IDENTIFIER_TLV
        self.egress_id_ui = egress_id_ui
        self.egress_id_mac = egress_id_mac

    @classmethod
    def parser(cls, buf):
        type_, length, egress_id_ui, egress_id_mac = struct.unpack_from(cls._PACK_STR, buf)
        return cls(length, egress_id_ui, addrconv.mac.bin_to_text(egress_id_mac))

    def serialize(self):
        if self.length == 0:
            self.length = 8
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.egress_id_ui, addrconv.mac.text_to_bin(self.egress_id_mac))
        return bytearray(buf)