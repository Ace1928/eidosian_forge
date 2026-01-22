import binascii
import struct
from . import packet_base
from os_ken.lib import addrconv
Rapid Spanning Tree BPDUs(RST BPDUs, IEEE 802.1D)
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
    