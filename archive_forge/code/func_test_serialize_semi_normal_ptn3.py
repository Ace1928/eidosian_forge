import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
def test_serialize_semi_normal_ptn3(self):
    ins = cfm.sender_id_tlv(chassis_id_subtype=self.chassis_id_subtype, chassis_id=self.chassis_id)
    buf = ins.serialize()
    form = '!BHBB1sB'
    res = struct.unpack_from(form, bytes(buf))
    self.assertEqual(self._type, res[0])
    self.assertEqual(4, res[1])
    self.assertEqual(self.chassis_id_length, res[2])
    self.assertEqual(self.chassis_id_subtype, res[3])
    self.assertEqual(self.chassis_id, res[4])
    self.assertEqual(0, res[5])