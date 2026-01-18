import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_with_length_zero(self):
    ins = cfm.ltr_egress_identifier_tlv(0, self.last_egress_id_ui, self.last_egress_id_mac, self.next_egress_id_ui, self.next_egress_id_mac)
    buf = ins.serialize()
    res = struct.unpack_from(self.form, bytes(buf))
    self.assertEqual(self._type, res[0])
    self.assertEqual(self.length, res[1])
    self.assertEqual(self.last_egress_id_ui, res[2])
    self.assertEqual(addrconv.mac.text_to_bin(self.last_egress_id_mac), res[3])
    self.assertEqual(self.next_egress_id_ui, res[4])
    self.assertEqual(addrconv.mac.text_to_bin(self.next_egress_id_mac), res[5])