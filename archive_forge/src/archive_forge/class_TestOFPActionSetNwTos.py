import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionSetNwTos(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionSetNwTos
    """
    type_ = {'buf': b'\x00\x08', 'val': ofproto.OFPAT_SET_NW_TOS}
    len_ = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_NW_TOS_SIZE}
    tos = {'buf': b'\xb6', 'val': 182}
    zfill = b'\x00' * 3
    buf = type_['buf'] + len_['buf'] + tos['buf'] + zfill
    c = OFPActionSetNwTos(tos['val'])

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.tos['val'], self.c.tos)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.tos['val'], res.tos)

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x05', 'val': 5}
        buf = type_['buf'] + self.len_['buf'] + self.tos['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x07', 'val': 7}
        buf = self.type_['buf'] + len_['buf'] + self.tos['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_serialize(self):
        buf = bytearray()
        self.c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_NW_TOS_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(self.type_['val'], res[0])
        self.assertEqual(self.len_['val'], res[1])
        self.assertEqual(self.tos['val'], res[2])