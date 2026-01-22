import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPTableMod(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPTableMod
    """
    table_id = 3
    config = 2226555987

    def test_init(self):
        c = OFPTableMod(_Datapath, self.table_id, self.config)
        self.assertEqual(self.table_id, c.table_id)
        self.assertEqual(self.config, c.config)

    def _test_serialize(self, table_id, config):
        c = OFPTableMod(_Datapath, table_id, config)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_TABLE_MOD, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_TABLE_MOD_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_TABLE_MOD)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], table_id)
        self.assertEqual(res[5], config)

    def test_serialize_mid(self):
        self._test_serialize(self.table_id, self.config)

    def test_serialize_max(self):
        table_id = ofproto.OFPTT_ALL
        config = 4294967295
        self._test_serialize(table_id, config)

    def test_serialize_min(self):
        table_id = 0
        config = 0
        self._test_serialize(table_id, config)

    def test_serialize_p1(self):
        table_id = ofproto.OFPTT_MAX
        config = ofproto.OFPTC_TABLE_MISS_CONTINUE
        self._test_serialize(table_id, config)

    def test_serialize_p2(self):
        table_id = ofproto.OFPTT_MAX
        config = ofproto.OFPTC_TABLE_MISS_DROP
        self._test_serialize(table_id, config)

    def test_serialize_p3(self):
        table_id = ofproto.OFPTT_MAX
        config = ofproto.OFPTC_TABLE_MISS_MASK
        self._test_serialize(table_id, config)