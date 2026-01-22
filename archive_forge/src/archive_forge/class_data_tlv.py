import abc
import struct
from os_ken.lib import addrconv
from os_ken.lib import stringify
from os_ken.lib.packet import packet_base
@operation.register_tlv_types(CFM_DATA_TLV)
class data_tlv(tlv):
    """CFM (IEEE Std 802.1ag-2007) Data TLV encoder/decoder class.

    This is used with os_ken.lib.packet.cfm.cfm.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    ============== =======================================
    Attribute      Description
    ============== =======================================
    length         Length of Value field.
                   (0 means automatically-calculate when encoding)
    data_value     Bit pattern of any of n octets.(n = length)
    ============== =======================================
    """
    _PACK_STR = '!BH'
    _MIN_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, length=0, data_value=b''):
        super(data_tlv, self).__init__(length)
        self._type = CFM_DATA_TLV
        self.data_value = data_value

    @classmethod
    def parser(cls, buf):
        type_, length = struct.unpack_from(cls._PACK_STR, buf)
        form = '%ds' % length
        data_value, = struct.unpack_from(form, buf, cls._MIN_LEN)
        return cls(length, data_value)

    def serialize(self):
        if self.length == 0:
            self.length = len(self.data_value)
        buf = struct.pack(self._PACK_STR, self._type, self.length)
        buf = bytearray(buf)
        form = '%ds' % self.length
        buf.extend(struct.pack(form, self.data_value))
        return buf