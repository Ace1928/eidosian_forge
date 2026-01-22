import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionOutputReg(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionOutputReg
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00\x18', 'val': ofproto.NX_ACTION_OUTPUT_REG_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': ofproto_common.NX_EXPERIMENTER_ID}
    subtype = {'buf': b'\x00\x0f', 'val': ofproto.NXAST_OUTPUT_REG}
    ofs_nbits = {'buf': b'\xfex', 'val': 65144}
    src = {'buf': b'\x00\x01\x00\x04', 'val': 'reg0', 'val2': 65540}
    max_len = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_OUTPUT_SIZE}
    zfill = b'\x00' * 6
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + ofs_nbits['buf'] + src['buf'] + max_len['buf'] + zfill
    c = NXActionOutputReg(ofs_nbits['val'], src['val'], max_len['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.ofs_nbits['val'], self.c.ofs_nbits)
        self.assertEqual(self.src['val'], self.c.src)
        self.assertEqual(self.max_len['val'], self.c.max_len)

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.subtype['val'], res.subtype)
        self.assertEqual(self.ofs_nbits['val'], self.c.ofs_nbits)
        self.assertEqual(self.src['val'], res.src)
        self.assertEqual(self.max_len['val'], res.max_len)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_OUTPUT_REG_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.ofs_nbits['val'], res[4])
        self.assertEqual(self.src['val2'], res[5])
        self.assertEqual(self.max_len['val'], res[6])