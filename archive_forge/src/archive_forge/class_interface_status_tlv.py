import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_INTERFACE_STATUS_TLV)
class interface_status_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) Interface Status TLV encoder/decoder class.

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
    interface_status     Interface Status.The default is 1 (isUp)
    ==================== =======================================
    """
    _PACK_STR = '!BHB'
    _MIN_LEN = struct.calcsize(_PACK_STR)
    _IS_UP = 1
    _IS_DOWN = 2
    _IS_TESTING = 3
    _IS_UNKNOWN = 4
    _IS_DORMANT = 5
    _IS_NOT_PRESENT = 6
    _IS_LOWER_LAYER_DOWN = 7

    def __init__(self, length=0, interface_status=_IS_UP):
        super(interface_status_tlv, self).__init__(length)
        self._type = CFM_INTERFACE_STATUS_TLV
        assert interface_status in [self._IS_UP, self._IS_DOWN, self._IS_TESTING, self._IS_UNKNOWN, self._IS_DORMANT, self._IS_NOT_PRESENT, self._IS_LOWER_LAYER_DOWN]
        self.interface_status = interface_status

    @classmethod
    def parser(cls, buf):
        type_, length, interface_status = struct.unpack_from(cls._PACK_STR, buf)
        return cls(length, interface_status)

    def serialize(self):
        if self.length == 0:
            self.length = 1
        buf = struct.pack(self._PACK_STR, self._type, self.length, self.interface_status)
        return bytearray(buf)