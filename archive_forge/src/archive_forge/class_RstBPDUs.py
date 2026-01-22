import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@bpdu.register_bpdu_type
class RstBPDUs(ConfigurationBPDUs):
    """Rapid Spanning Tree BPDUs(RST BPDUs, IEEE 802.1D)
    header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===========================================
    Attribute                  Description
    ========================== ===========================================
    flags                      | Bit 1: Topology Change flag
                               | Bit 2: Proposal flag
                               | Bits 3 and 4: Port Role
                               | Bit 5: Learning flag
                               | Bit 6: Forwarding flag
                               | Bit 7: Agreement flag
                               | Bit 8: Topology Change Acknowledgment flag
    root_priority              Root Identifier priority                                set 0-61440 in steps of 4096
    root_system_id_extension   Root Identifier system ID extension
    root_mac_address           Root Identifier MAC address
    root_path_cost             Root Path Cost
    bridge_priority            Bridge Identifier priority                                set 0-61440 in steps of 4096
    bridge_system_id_extension Bridge Identifier system ID extension
    bridge_mac_address         Bridge Identifier MAC address
    port_priority              Port Identifier priority                                set 0-240 in steps of 16
    port_number                Port Identifier number
    message_age                Message Age timer value
    max_age                    Max Age timer value
    hello_time                 Hello Time timer value
    forward_delay              Forward Delay timer value
    ========================== ===========================================
    """
    VERSION_ID = PROTOCOLVERSION_ID_RSTBPDU
    BPDU_TYPE = TYPE_RSTBPDU
    _PACK_STR = '!B'
    PACK_LEN = struct.calcsize(_PACK_STR)

    def __init__(self, flags=0, root_priority=DEFAULT_BRIDGE_PRIORITY, root_system_id_extension=0, root_mac_address='00:00:00:00:00:00', root_path_cost=0, bridge_priority=DEFAULT_BRIDGE_PRIORITY, bridge_system_id_extension=0, bridge_mac_address='00:00:00:00:00:00', port_priority=DEFAULT_PORT_PRIORITY, port_number=0, message_age=0, max_age=DEFAULT_MAX_AGE, hello_time=DEFAULT_HELLO_TIME, forward_delay=DEFAULT_FORWARD_DELAY):
        self._version_1_length = VERSION_1_LENGTH
        super(RstBPDUs, self).__init__(flags, root_priority, root_system_id_extension, root_mac_address, root_path_cost, bridge_priority, bridge_system_id_extension, bridge_mac_address, port_priority, port_number, message_age, max_age, hello_time, forward_delay)

    def check_parameters(self):
        assert self.root_priority % self._BRIDGE_PRIORITY_STEP == 0
        assert self.bridge_priority % self._BRIDGE_PRIORITY_STEP == 0
        assert self.port_priority % self._PORT_PRIORITY_STEP == 0
        assert self.message_age % self._TIMER_STEP == 0
        assert self.max_age % self._TIMER_STEP == 0
        assert self.hello_time % self._TIMER_STEP == 0
        assert self.forward_delay % self._TIMER_STEP == 0

    @classmethod
    def parser(cls, buf):
        get_cls, next_type, buf = super(RstBPDUs, cls).parser(buf)
        version_1_length, = struct.unpack_from(RstBPDUs._PACK_STR, buf)
        assert version_1_length == VERSION_1_LENGTH
        return (get_cls, next_type, buf[RstBPDUs.PACK_LEN:])

    def serialize(self, payload, prev):
        base = super(RstBPDUs, self).serialize(payload, prev)
        sub = struct.pack(RstBPDUs._PACK_STR, self._version_1_length)
        return base + sub