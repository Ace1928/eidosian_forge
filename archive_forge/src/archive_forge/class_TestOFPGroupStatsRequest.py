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
class TestOFPGroupStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGroupStatsRequest
    """
    group_id = 6606

    def test_init(self):
        c = OFPGroupStatsRequest(_Datapath, self.group_id)
        self.assertEqual(self.group_id, c.group_id)

    def _test_serialize(self, group_id):
        c = OFPGroupStatsRequest(_Datapath, group_id)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_STATS_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_STATS_REQUEST_PACK_STR.replace('!', '') + ofproto.OFP_GROUP_STATS_REQUEST_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_STATS_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], ofproto.OFPST_GROUP)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], group_id)

    def test_serialize_mid(self):
        self._test_serialize(self.group_id)

    def test_serialize_max(self):
        self._test_serialize(4294967295)

    def test_serialize_min(self):
        self._test_serialize(0)

    def test_serialize_p1(self):
        self._test_serialize(ofproto.OFPG_MAX)

    def test_serialize_p2(self):
        self._test_serialize(ofproto.OFPG_ALL)