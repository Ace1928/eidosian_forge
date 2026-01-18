import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_zero(self):
    ins = cfm.reply_egress_tlv(0, self.action, self.mac_address, 0, self.port_id_subtype, self.port_id)
    buf = ins.serialize()
    res = struct.unpack_from(self.form, bytes(buf))
    self.assertEqual(self._type, res[0])
    self.assertEqual(self.length, res[1])
    self.assertEqual(self.action, res[2])
    self.assertEqual(addrconv.mac.text_to_bin(self.mac_address), res[3])
    self.assertEqual(self.port_id_length, res[4])
    self.assertEqual(self.port_id_subtype, res[5])
    self.assertEqual(self.port_id, res[6])