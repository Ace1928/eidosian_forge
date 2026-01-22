import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_LTR_EGRESS_IDENTIFIER_TLV)
class ltr_egress_identifier_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) LTR EGRESS TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ==================== =======================================
    Attribute            Description
    ==================== =======================================
    length               Length of Value field.
                         (0 means automatically-calculate when encoding.)
    last_egress_id_ui    Last Egress Identifier of Unique ID.
    last_egress_id_mac   Last Egress Identifier of MAC address.
    next_egress_id_ui    Next Egress Identifier of Unique ID.
    next_egress_id_mac   Next Egress Identifier of MAC address.
    ==================== =======================================
    """
    _PACK_STR = '!BHH6sH6s'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, length=0, last_egress_id_ui=0, last_egress_id_mac='00:00:00:00:00:00', next_egress_id_ui=0, next_egress_id_mac='00:00:00:00:00:00'):
        super(ltr_egress_identifier_tlv, self).__init__(length)
        self._type = CFM_LTR_EGRESS_IDENTIFIER_TLV
        self.last_egress_id_ui = last_egress_id_ui
        self.last_egress_id_mac = last_egress_id_mac
        self.next_egress_id_ui = next_egress_id_ui
        self.next_egress_id_mac = next_egress_id_mac

    @classmethod
    def parser(cls, buf):
        type_, length, last_egress_id_ui, last_egress_id_mac, next_egress_id_ui, next_egress_id_mac = struct.unpack_from(cls._PACK_STR, buf)
        return cls(length, last_egress_id_ui, addrconv.mac.bin_to_text(last_egress_id_mac), next_egress_id_ui, addrconv.mac.bin_to_text(next_egress_id_mac))

    def serialize(self):
        if self.length == 0:
            self.length = 16
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.last_egress_id_ui, addrconv.mac.text_to_bin(self.last_egress_id_mac), self.next_egress_id_ui, addrconv.mac.text_to_bin(self.next_egress_id_mac))
        return bytearray(buf)