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
class TestOFPAggregateStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPAggregateStatsRequest
    """
    table_id = 3
    out_port = 65037
    out_group = 6606
    cookie = 2127614848199081640
    cookie_mask = 2127614848199081641

    def test_init(self):
        match = OFPMatch()
        dl_type = 2048
        match.set_dl_type(dl_type)
        c = OFPAggregateStatsRequest(_Datapath, self.table_id, self.out_port, self.out_group, self.cookie, self.cookie_mask, match)
        self.assertEqual(self.table_id, c.table_id)
        self.assertEqual(self.out_port, c.out_port)
        self.assertEqual(self.out_group, c.out_group)
        self.assertEqual(self.cookie, c.cookie)
        self.assertEqual(self.cookie_mask, c.cookie_mask)
        self.assertEqual(dl_type, c.match._flow.dl_type)

    def _test_serialize(self, table_id, out_port, out_group, cookie, cookie_mask):
        match = OFPMatch()
        dl_type = 2048
        match.set_dl_type(dl_type)
        c = OFPAggregateStatsRequest(_Datapath, table_id, out_port, out_group, cookie, cookie_mask, match)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_STATS_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR + ofproto.OFP_STATS_REQUEST_PACK_STR[1:] + ofproto.OFP_AGGREGATE_STATS_REQUEST_PACK_STR[1:] + 'HHHBB' + MTEthType.pack_str[1:] + '6x'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_STATS_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], ofproto.OFPST_AGGREGATE)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], table_id)
        self.assertEqual(res[7], out_port)
        self.assertEqual(res[8], out_group)
        self.assertEqual(res[9], cookie)
        self.assertEqual(res[10], cookie_mask)
        self.assertEqual(res[11], ofproto.OFPMT_OXM)
        self.assertEqual(res[12], 10)
        self.assertEqual(res[13], ofproto.OFPXMC_OPENFLOW_BASIC)
        self.assertEqual(res[14] >> 1, ofproto.OFPXMT_OFB_ETH_TYPE)
        self.assertEqual(res[14] & 1, 0)
        self.assertEqual(res[15], calcsize(MTEthType.pack_str))
        self.assertEqual(res[16], dl_type)

    def test_serialize_mid(self):
        self._test_serialize(self.table_id, self.out_port, self.out_group, self.cookie, self.cookie_mask)

    def test_serialize_max(self):
        table_id = 255
        out_port = 4294967295
        out_group = 4294967295
        cookie = 4294967295
        cookie_mask = 4294967295
        self._test_serialize(table_id, out_port, out_group, cookie, cookie_mask)

    def test_serialize_min(self):
        table_id = 0
        out_port = 0
        out_group = 0
        cookie = 0
        cookie_mask = 0
        self._test_serialize(table_id, out_port, out_group, cookie, cookie_mask)

    def test_serialize_p1(self):
        table_id = ofproto.OFPTT_MAX
        self._test_serialize(table_id, self.out_port, self.out_group, self.cookie, self.cookie_mask)