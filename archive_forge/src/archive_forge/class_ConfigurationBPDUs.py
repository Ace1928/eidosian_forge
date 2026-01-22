import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
@bpdu.register_bpdu_type
class ConfigurationBPDUs(bpdu):
    """Configuration BPDUs(IEEE 802.1D) header encoder/decoder class.

    An instance has the following attributes at least.
    Most of them are same to the on-wire counterparts but in host byte
    order.
    __init__ takes the corresponding args in this order.

    ========================== ===============================================
    Attribute                  Description
    ========================== ===============================================
    flags                      | Bit 1: Topology Change flag
                               | Bits 2 through 7: unused and take the value 0
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
    ========================== ===============================================
    """
    VERSION_ID = PROTOCOLVERSION_ID_BPDU
    BPDU_TYPE = TYPE_CONFIG_BPDU
    _PACK_STR = '!BQIQHHHHH'
    PACK_LEN = struct.calcsize(_PACK_STR)
    _TYPE = {'ascii': ['root_mac_address', 'bridge_mac_address']}
    _BRIDGE_PRIORITY_STEP = 4096
    _PORT_PRIORITY_STEP = 16
    _TIMER_STEP = float(1) / 256

    def __init__(self, flags=0, root_priority=DEFAULT_BRIDGE_PRIORITY, root_system_id_extension=0, root_mac_address='00:00:00:00:00:00', root_path_cost=0, bridge_priority=DEFAULT_BRIDGE_PRIORITY, bridge_system_id_extension=0, bridge_mac_address='00:00:00:00:00:00', port_priority=DEFAULT_PORT_PRIORITY, port_number=0, message_age=0, max_age=DEFAULT_MAX_AGE, hello_time=DEFAULT_HELLO_TIME, forward_delay=DEFAULT_FORWARD_DELAY):
        self.flags = flags
        self.root_priority = root_priority
        self.root_system_id_extension = root_system_id_extension
        self.root_mac_address = root_mac_address
        self.root_path_cost = root_path_cost
        self.bridge_priority = bridge_priority
        self.bridge_system_id_extension = bridge_system_id_extension
        self.bridge_mac_address = bridge_mac_address
        self.port_priority = port_priority
        self.port_number = port_number
        self.message_age = message_age
        self.max_age = max_age
        self.hello_time = hello_time
        self.forward_delay = forward_delay
        super(ConfigurationBPDUs, self).__init__()

    def check_parameters(self):
        assert self.flags >> 1 & 63 == 0
        assert self.root_priority % self._BRIDGE_PRIORITY_STEP == 0
        assert self.bridge_priority % self._BRIDGE_PRIORITY_STEP == 0
        assert self.port_priority % self._PORT_PRIORITY_STEP == 0
        assert self.message_age % self._TIMER_STEP == 0
        assert self.max_age % self._TIMER_STEP == 0
        assert self.hello_time % self._TIMER_STEP == 0
        assert self.forward_delay % self._TIMER_STEP == 0

    @classmethod
    def parser(cls, buf):
        flags, root_id, root_path_cost, bridge_id, port_id, message_age, max_age, hello_time, forward_delay = struct.unpack_from(ConfigurationBPDUs._PACK_STR, buf)
        root_priority, root_system_id_extension, root_mac_address = cls._decode_bridge_id(root_id)
        bridge_priority, bridge_system_id_extension, bridge_mac_address = cls._decode_bridge_id(bridge_id)
        port_priority, port_number = cls._decode_port_id(port_id)
        return (cls(flags, root_priority, root_system_id_extension, root_mac_address, root_path_cost, bridge_priority, bridge_system_id_extension, bridge_mac_address, port_priority, port_number, cls._decode_timer(message_age), cls._decode_timer(max_age), cls._decode_timer(hello_time), cls._decode_timer(forward_delay)), None, buf[ConfigurationBPDUs.PACK_LEN:])

    def serialize(self, payload, prev):
        base = super(ConfigurationBPDUs, self).serialize(payload, prev)
        root_id = self.encode_bridge_id(self.root_priority, self.root_system_id_extension, self.root_mac_address)
        bridge_id = self.encode_bridge_id(self.bridge_priority, self.bridge_system_id_extension, self.bridge_mac_address)
        port_id = self.encode_port_id(self.port_priority, self.port_number)
        sub = struct.pack(ConfigurationBPDUs._PACK_STR, self.flags, root_id, self.root_path_cost, bridge_id, port_id, self._encode_timer(self.message_age), self._encode_timer(self.max_age), self._encode_timer(self.hello_time), self._encode_timer(self.forward_delay))
        return base + sub

    @staticmethod
    def _decode_bridge_id(bridge_id):
        priority = bridge_id >> 48 & 61440
        system_id_extension = bridge_id >> 48 & 4095
        mac_addr = bridge_id & 281474976710655
        mac_addr_list = [format(mac_addr >> 8 * i & 255, '02x') for i in range(0, 6)]
        mac_addr_list.reverse()
        mac_address_bin = binascii.a2b_hex(''.join(mac_addr_list))
        mac_address = addrconv.mac.bin_to_text(mac_address_bin)
        return (priority, system_id_extension, mac_address)

    @staticmethod
    def encode_bridge_id(priority, system_id_extension, mac_address):
        mac_addr = int(binascii.hexlify(addrconv.mac.text_to_bin(mac_address)), 16)
        return (priority + system_id_extension << 48) + mac_addr

    @staticmethod
    def _decode_port_id(port_id):
        priority = port_id >> 8 & 240
        port_number = port_id & 4095
        return (priority, port_number)

    @staticmethod
    def encode_port_id(priority, port_number):
        return (priority << 8) + port_number

    @staticmethod
    def _decode_timer(timer):
        return timer / float(256)

    @staticmethod
    def _encode_timer(timer):
        return int(timer) * 256