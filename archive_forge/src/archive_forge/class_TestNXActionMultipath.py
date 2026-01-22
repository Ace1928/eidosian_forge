import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestNXActionMultipath(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.NXActionMultipath
    """
    type_ = {'buf': b'\xff\xff', 'val': ofproto.OFPAT_VENDOR}
    len_ = {'buf': b'\x00 ', 'val': ofproto.NX_ACTION_MULTIPATH_SIZE}
    vendor = {'buf': b'\x00\x00# ', 'val': ofproto_common.NX_EXPERIMENTER_ID}
    subtype = {'buf': b'\x00\n', 'val': ofproto.NXAST_MULTIPATH}
    fields = {'buf': b'm\xf5', 'val': 28149}
    basis = {'buf': b'|\n', 'val': 31754}
    zfill0 = b'\x00' * 2
    algorithm = {'buf': b'\x82\x1d', 'val': 33309}
    max_link = {'buf': b'\x06+', 'val': 1579}
    arg = {'buf': b'\x18yA\xc8', 'val': 410599880}
    zfill1 = b'\x00' * 2
    ofs_nbits = {'buf': b'\xa9\x9a', 'val': 43418}
    dst = {'buf': b'\x00\x01\x00\x04', 'val': 'reg0', 'val2': 65540}
    start = 678
    end = 704
    buf = type_['buf'] + len_['buf'] + vendor['buf'] + subtype['buf'] + fields['buf'] + basis['buf'] + zfill0 + algorithm['buf'] + max_link['buf'] + arg['buf'] + zfill1 + ofs_nbits['buf'] + dst['buf']
    c = NXActionMultipath(fields['val'], basis['val'], algorithm['val'], max_link['val'], arg['val'], ofs_nbits['val'], dst['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.subtype['val'], self.c.subtype)
        self.assertEqual(self.fields['val'], self.c.fields)
        self.assertEqual(self.basis['val'], self.c.basis)
        self.assertEqual(self.algorithm['val'], self.c.algorithm)
        self.assertEqual(self.max_link['val'], self.c.max_link)
        self.assertEqual(self.arg['val'], self.c.arg)
        self.assertEqual(self.ofs_nbits['val'], self.c.ofs_nbits)
        self.assertEqual(self.dst['val'], self.c.dst)

    def test_parser(self):
        res = OFPActionVendor.parser(self.buf, 0)
        self.assertEqual(self.type_['val'], res.type)
        self.assertEqual(self.len_['val'], res.len)
        self.assertEqual(self.subtype['val'], res.subtype)
        self.assertEqual(self.fields['val'], res.fields)
        self.assertEqual(self.basis['val'], res.basis)
        self.assertEqual(self.algorithm['val'], res.algorithm)
        self.assertEqual(self.max_link['val'], res.max_link)
        self.assertEqual(self.arg['val'], res.arg)
        self.assertEqual(self.ofs_nbits['val'], res.ofs_nbits)
        self.assertEqual(self.dst['val'], res.dst)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.NX_ACTION_MULTIPATH_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.vendor['val'], res[2])
        self.assertEqual(self.subtype['val'], res[3])
        self.assertEqual(self.fields['val'], res[4])
        self.assertEqual(self.basis['val'], res[5])
        self.assertEqual(self.algorithm['val'], res[6])
        self.assertEqual(self.max_link['val'], res[7])
        self.assertEqual(self.arg['val'], res[8])
        self.assertEqual(self.ofs_nbits['val'], res[9])
        self.assertEqual(self.dst['val2'], res[10])