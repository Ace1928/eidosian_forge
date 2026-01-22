import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_REPLY_EGRESS_TLV)
class reply_egress_tlv(reply_tlv):
    """CFM (IEEE Std 802.1ag-2007) Reply Egress TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ================= =======================================
    Attribute         Description
    ================= =======================================
    length            Length of Value field.
                      (0 means automatically-calculate when encoding.)
    action            Egress Action.The default is 1 (EgrOK)
    mac_address       Egress MAC Address.
    port_id_length    Egress PortID Length.
                      (0 means automatically-calculate when encoding.)
    port_id_subtype   Egress PortID Subtype.
    port_id           Egress PortID.
    ================= =======================================
    """
    _EGR_OK = 1
    _EGR_DOWN = 2
    _EGR_BLOCKED = 3
    _EGR_VID = 4

    def __init__(self, length=0, action=_EGR_OK, mac_address='00:00:00:00:00:00', port_id_length=0, port_id_subtype=0, port_id=b''):
        super(reply_egress_tlv, self).__init__(length, action, mac_address, port_id_length, port_id_subtype, port_id)
        assert action in [self._EGR_OK, self._EGR_DOWN, self._EGR_BLOCKED, self._EGR_VID]
        self._type = CFM_REPLY_EGRESS_TLV