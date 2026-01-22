import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_port_status_tlv(unittest.TestCase):

    def setUp(self):
        self._type = cfm.CFM_PORT_STATUS_TLV
        self.length = 1
        self.port_status = 1
        self.ins = cfm.port_status_tlv(self.length, self.port_status)
        self.form = '!BHB'
        self.buf = struct.pack(self.form, self._type, self.length, self.port_status)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.length, self.ins.length)
        self.assertEqual(self.port_status, self.ins.port_status)

    def test_parser(self):
        _res = cfm.port_status_tlv.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.length, res.length)
        self.assertEqual(self.port_status, res.port_status)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self._type, res[0])
        self.assertEqual(self.length, res[1])
        self.assertEqual(self.port_status, res[2])

    def test_len(self):
        self.assertEqual(1 + 2 + 1, len(self.ins))

    def test_default_args(self):
        ins = cfm.port_status_tlv()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.port_status_tlv._PACK_STR, bytes(buf))
        self.assertEqual(res[0], cfm.CFM_PORT_STATUS_TLV)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 2)