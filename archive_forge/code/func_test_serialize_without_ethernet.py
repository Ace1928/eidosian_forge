import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import lldp
from os_ken.lib import addrconv
def test_serialize_without_ethernet(self):
    tlv_chassis_id = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
    tlv_port_id = lldp.PortID(subtype=lldp.PortID.SUB_INTERFACE_NAME, port_id=b'1/3')
    tlv_ttl = lldp.TTL(ttl=120)
    tlv_end = lldp.End()
    tlvs = (tlv_chassis_id, tlv_port_id, tlv_ttl, tlv_end)
    lldp_pkt = lldp.lldp(tlvs)
    self.assertEqual(lldp_pkt.serialize(None, None), self.data[ethernet.ethernet._MIN_LEN:])