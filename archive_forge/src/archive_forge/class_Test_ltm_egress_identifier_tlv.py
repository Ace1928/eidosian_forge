import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_ltm_egress_identifier_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_LTM_EGRESS_IDENTIFIER_TLV
        self.length = 8
        self.egress_id_ui = 7
        self.egress_id_mac = '11:22:33:44:55:66'
        self.ins = cfm.ltm_egress_identifier_tlv(self.length, self.egress_id_ui, self.egress_id_mac)
        self.form = '!BHH6s'
        self.buf = struct.pack(self.form, self._type, self.length, self.egress_id_ui, addrconv.mac.text_to_bin(self.egress_id_mac))

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.egress_id_ui, self.ins.egress_id_ui)
        self.assertEqual(self.egress_id_mac, self.ins.egress_id_mac)

    def test_parser(self):
        _res = cfm.ltm_egress_identifier_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.egress_id_ui, res.egress_id_ui)
        self.assertEqual(self.egress_id_mac, res.egress_id_mac)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.egress_id_ui, res[2])
        self.assertEqual(addrconv.mac.text_to_bin(self.egress_id_mac), res[3])

    def test_serialize_with_length_zero(self):
        ins = cfm.ltm_egress_identifier_tlv(0, self.egress_id_ui, self.egress_id_mac)
        buf = ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.egress_id_ui, res[2])
        self.assertEqual(addrconv.mac.text_to_bin(self.egress_id_mac), res[3])

    def test_len(self):
        self.assertEqual(1 + 2 + 8, len(self.ins))

    def test_default_args(self):
        ins = cfm.ltm_egress_identifier_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.ltm_egress_identifier_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_LTM_EGRESS_IDENTIFIER_TLV)
        self.assertEqual(res[1], 8)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.mac.text_to_bin('00:00:00:00:00:00'))