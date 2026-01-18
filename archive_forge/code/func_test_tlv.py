import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import lldp
from os_ken.lib import addrconv
def test_tlv(self):
    tlv = lldp.ChassisID(subtype=lldp.ChassisID.SUB_MAC_ADDRESS, chassis_id=b'\x00\x04\x96\x1f\xa7&')
    self.assertEqual(tlv.tlv_type, lldp.LLDP_TLV_CHASSIS_ID)
    self.assertEqual(tlv.len, 7)
    typelen, = struct.unpack('!H', b'\x02\x07')
    self.assertEqual(tlv.typelen, typelen)