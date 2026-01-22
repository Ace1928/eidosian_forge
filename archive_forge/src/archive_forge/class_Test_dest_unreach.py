import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
class Test_dest_unreach(unittest.TestCase):

    def setUp(self):
        self.mtu = 10
        self.data = b'abc'
        self.data_len = len(self.data)
        self.dest_unreach = icmp.dest_unreach(data_len=self.data_len, mtu=self.mtu, data=self.data)
        self.buf = struct.pack('!xBH', self.data_len, self.mtu)
        self.buf += self.data

    def test_init(self):
        self.assertEqual(self.data_len, self.dest_unreach.data_len)
        self.assertEqual(self.mtu, self.dest_unreach.mtu)
        self.assertEqual(self.data, self.dest_unreach.data)

    def test_parser(self):
        _res = icmp.dest_unreach.parser(self.buf, 0)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.data_len, res.data_len)
        self.assertEqual(self.mtu, res.mtu)
        self.assertEqual(self.data, res.data)

    def test_serialize(self):
        buf = self.dest_unreach.serialize()
        res = struct.unpack_from('!xBH', bytes(buf))
        self.assertEqual(self.data_len, res[0])
        self.assertEqual(self.mtu, res[1])
        self.assertEqual(self.data, buf[struct.calcsize('!xBH'):])

    def test_default_args(self):
        du = icmp.dest_unreach()
        buf = du.serialize()
        res = struct.unpack(icmp.dest_unreach._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)