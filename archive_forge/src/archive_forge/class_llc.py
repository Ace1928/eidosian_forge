import struct
from . import bpdu
from . import packet_base
from os_ken.lib import stringify
class llc(packet_base.PacketBase):
    """LLC(IEEE 802.2) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    .. tabularcolumns:: |l|L|

    =============== ===============================================
    Attribute       Description
    =============== ===============================================
    dsap_addr       Destination service access point address field                     includes I/G bit at least significant bit.
    ssap_addr       Source service access point address field                     includes C/R bit at least significant bit.
    control         Control field                     [16 bits for formats that include sequence                     numbering, and 8 bits for formats that do not].                     Either os_ken.lib.packet.llc.ControlFormatI or                     os_ken.lib.packet.llc.ControlFormatS or                     os_ken.lib.packet.llc.ControlFormatU object.
    =============== ===============================================
    """
    _PACK_STR = '!BB'
    _PACK_LEN = struct.calcsize(_PACK_STR)
    _CTR_TYPES = {}
    _CTR_PACK_STR = '!2xB'
    _MIN_LEN = _PACK_LEN

    @staticmethod
    def register_control_type(register_cls):
        llc._CTR_TYPES[register_cls.TYPE] = register_cls
        return register_cls

    def __init__(self, dsap_addr, ssap_addr, control):
        super(llc, self).__init__()
        assert getattr(control, 'TYPE', None) in self._CTR_TYPES
        self.dsap_addr = dsap_addr
        self.ssap_addr = ssap_addr
        self.control = control

    @classmethod
    def parser(cls, buf):
        assert len(buf) >= cls._PACK_LEN
        dsap_addr, ssap_addr = struct.unpack_from(cls._PACK_STR, buf)
        control, = struct.unpack_from(cls._CTR_PACK_STR, buf)
        ctrl = cls._get_control(control)
        control, information = ctrl.parser(buf[cls._PACK_LEN:])
        return (cls(dsap_addr, ssap_addr, control), cls.get_packet_type(dsap_addr), information)

    def serialize(self, payload, prev):
        addr = struct.pack(self._PACK_STR, self.dsap_addr, self.ssap_addr)
        control = self.control.serialize()
        return addr + control

    @classmethod
    def _get_control(cls, buf):
        key = buf & 1 if buf & 1 == ControlFormatI.TYPE else buf & 3
        return cls._CTR_TYPES[key]