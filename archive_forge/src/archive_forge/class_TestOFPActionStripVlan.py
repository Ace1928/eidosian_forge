import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
class TestOFPActionStripVlan(unittest.TestCase):
    """ Test case for ofproto_v1_0_parser.OFPActionStripVlan
    """
    type_ = {'buf': b'\x00\x03', 'val': ofproto.OFPAT_STRIP_VLAN}
    len_ = {'buf': b'\x00\x08', 'val': ofproto.OFP_ACTION_HEADER_SIZE}
    zfill = b'\x00' * 4
    buf = type_['buf'] + len_['buf'] + zfill
    c = OFPActionStripVlan()

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        pass

    def test_parser(self):
        self.assertTrue(self.c.parser(self.buf, 0))

    def test_parser_check_type(self):
        type_ = {'buf': b'\x00\x01', 'val': 1}
        buf = type_['buf'] + self.len_['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)

    def test_parser_check_len(self):
        len_ = {'buf': b'\x00\x07', 'val': 7}
        buf = self.type_['buf'] + len_['buf'] + self.zfill
        self.assertRaises(AssertionError, self.c.parser, buf, 0)