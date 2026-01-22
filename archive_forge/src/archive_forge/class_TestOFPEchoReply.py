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
class TestOFPEchoReply(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPEchoReply
    """
    version = ofproto.OFP_VERSION
    msg_type = ofproto.OFPT_ECHO_REPLY
    msg_len = ofproto.OFP_HEADER_SIZE
    xid = 2495926989

    def test_init(self):
        c = OFPEchoReply(_Datapath)
        self.assertEqual(c.data, None)

    def _test_parser(self, data):
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, self.version, self.msg_type, self.msg_len, self.xid)
        if data is not None:
            buf += data
        res = OFPEchoReply.parser(object, self.version, self.msg_type, self.msg_len, self.xid, buf)
        self.assertEqual(res.version, self.version)
        self.assertEqual(res.msg_type, self.msg_type)
        self.assertEqual(res.msg_len, self.msg_len)
        self.assertEqual(res.xid, self.xid)
        if data is not None:
            self.assertEqual(res.data, data)

    def test_parser_mid(self):
        data = b'Reply Message.'
        self._test_parser(data)

    def test_parser_max(self):
        data = b'Reply Message.'.ljust(65527)
        self._test_parser(data)

    def test_parser_min(self):
        data = None
        self._test_parser(data)

    def _test_serialize(self, data):
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, self.version, self.msg_type, self.msg_len, self.xid) + data
        c = OFPEchoReply(_Datapath)
        c.data = data
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_ECHO_REPLY, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + str(len(c.data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_ECHO_REPLY)
        self.assertEqual(res[2], len(buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], data)

    def test_serialize_mid(self):
        data = b'Reply Message.'
        self._test_serialize(data)

    def test_serialize_max(self):
        data = b'Reply Message.'.ljust(65527)
        self._test_serialize(data)

    def test_serialize_check_data(self):
        c = OFPEchoReply(_Datapath)
        self.assertRaises(AssertionError, c.serialize)