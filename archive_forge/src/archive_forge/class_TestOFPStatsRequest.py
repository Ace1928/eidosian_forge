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
class TestOFPStatsRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPStatsRequest
    """
    type_ = ofproto.OFPST_DESC
    c = OFPStatsRequest(_Datapath, type_)

    def test_init(self):
        self.assertEqual(self.type_, self.c.type)
        self.assertEqual(0, self.c.flags)

    def test_serialize_body(self):
        len_ = ofproto.OFP_HEADER_SIZE + ofproto.OFP_STATS_REQUEST_SIZE
        self.c.buf = bytearray(len_)
        self.c._serialize_body()
        fmt = ofproto.OFP_STATS_REQUEST_PACK_STR
        res = struct.unpack_from(fmt, bytes(self.c.buf), ofproto.OFP_HEADER_SIZE)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], 0)