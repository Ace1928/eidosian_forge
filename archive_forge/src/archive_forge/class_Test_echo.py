import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
class Test_echo(unittest.TestCase):

    def setUp(self):
        self.id_ = 13379
        self.seq = 1
        self.data = b'0\x0e\t\x00\x00\x00\x00\x00' + b'\x10\x11\x12\x13\x14\x15\x16\x17' + b'\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f' + b' !"#$%&\'' + b'()*+,-./' + b'01234567'
        self.echo = icmp.echo(self.id_, self.seq, self.data)
        self.buf = struct.pack('!HH', self.id_, self.seq)
        self.buf += self.data

    def test_init(self):
        self.assertEqual(self.id_, self.echo.id)
        self.assertEqual(self.seq, self.echo.seq)
        self.assertEqual(self.data, self.echo.data)

    def test_parser(self):
        _res = icmp.echo.parser(self.buf, 0)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.id_, res.id)
        self.assertEqual(self.seq, res.seq)
        self.assertEqual(self.data, res.data)

    def test_serialize(self):
        buf = self.echo.serialize()
        res = struct.unpack_from('!HH', bytes(buf))
        self.assertEqual(self.id_, res[0])
        self.assertEqual(self.seq, res[1])
        self.assertEqual(self.data, buf[struct.calcsize('!HH'):])

    def test_default_args(self):
        ec = icmp.echo()
        buf = ec.serialize()
        res = struct.unpack(icmp.echo._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)