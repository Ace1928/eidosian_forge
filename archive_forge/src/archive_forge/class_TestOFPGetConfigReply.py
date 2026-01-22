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
class TestOFPGetConfigReply(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPGetConfigReply
    """

    def _test_parser(self, xid, flags, miss_send_len):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_GET_CONFIG_REPLY
        msg_len = ofproto.OFP_SWITCH_CONFIG_SIZE
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_SWITCH_CONFIG_PACK_STR
        buf += pack(fmt, flags, miss_send_len)
        res = OFPGetConfigReply.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(flags, res.flags)
        self.assertEqual(miss_send_len, res.miss_send_len)

    def test_parser_mid(self):
        xid = 3423224276
        flags = 41186
        miss_send_len = 13838
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_max(self):
        xid = 4294967295
        flags = 65535
        miss_send_len = 65535
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_min(self):
        xid = 0
        flags = ofproto.OFPC_FRAG_NORMAL
        miss_send_len = 0
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_p1(self):
        xid = 3423224276
        flags = ofproto.OFPC_FRAG_DROP
        miss_send_len = 13838
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_p2(self):
        xid = 3423224276
        flags = ofproto.OFPC_FRAG_REASM
        miss_send_len = 13838
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_p3(self):
        xid = 3423224276
        flags = ofproto.OFPC_FRAG_MASK
        miss_send_len = 13838
        self._test_parser(xid, flags, miss_send_len)

    def test_parser_p4(self):
        xid = 3423224276
        flags = ofproto.OFPC_INVALID_TTL_TO_CONTROLLER
        miss_send_len = 13838
        self._test_parser(xid, flags, miss_send_len)