import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
class Test_TimeExceeded(unittest.TestCase):

    def setUp(self):
        self.data = b'abc'
        self.data_len = len(self.data)
        self.te = icmp.TimeExceeded(data_len=self.data_len, data=self.data)
        self.buf = struct.pack('!xBxx', self.data_len)
        self.buf += self.data

    def test_init(self):
        self.assertEqual(self.data_len, self.te.data_len)
        self.assertEqual(self.data, self.te.data)

    def test_parser(self):
        _res = icmp.TimeExceeded.parser(self.buf, 0)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.data_len, res.data_len)
        self.assertEqual(self.data, res.data)

    def test_serialize(self):
        buf = self.te.serialize()
        res = struct.unpack_from('!xBxx', bytes(buf))
        self.assertEqual(self.data_len, res[0])
        self.assertEqual(self.data, buf[struct.calcsize('!xBxx'):])

    def test_default_args(self):
        te = icmp.TimeExceeded()
        buf = te.serialize()
        res = struct.unpack(icmp.TimeExceeded._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)