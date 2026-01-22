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
class TestOFPEchoRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPEchoRequest
    """
    version = ofproto.OFP_VERSION
    msg_type = ofproto.OFPT_ECHO_REQUEST
    msg_len = ofproto.OFP_HEADER_SIZE
    xid = 2495926989

    def test_init(self):
        c = OFPEchoRequest(_Datapath)
        self.assertEqual(c.data, None)

    def _test_parser(self, data=None):
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, self.version, self.msg_type, self.msg_len, self.xid)
        if data is not None:
            buf += data
        res = OFPEchoRequest.parser(object, self.version, self.msg_type, self.msg_len, self.xid, buf)
        self.assertEqual(res.version, self.version)
        self.assertEqual(res.msg_type, self.msg_type)
        self.assertEqual(res.msg_len, self.msg_len)
        self.assertEqual(res.xid, self.xid)
        if data is not None:
            self.assertEqual(res.data, data)

    def test_parser_mid(self):
        data = b'Request Message.'
        self._test_parser(data)

    def test_parser_max(self):
        data = b'Request Message.'.ljust(65527)
        self._test_parser(data)

    def test_parser_min(self):
        data = None
        self._test_parser(data)

    def _test_serialize(self, data):
        c = OFPEchoRequest(_Datapath)
        c.data = data
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_ECHO_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = ofproto.OFP_HEADER_PACK_STR
        if data is not None:
            fmt += str(len(c.data)) + 's'
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_ECHO_REQUEST)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        if data is not None:
            self.assertEqual(res[4], data)

    def test_serialize_mid(self):
        data = b'Request Message.'
        self._test_serialize(data)

    def test_serialize_max(self):
        data = b'Request Message.'.ljust(65527)
        self._test_serialize(data)

    def test_serialize_min(self):
        data = None
        self._test_serialize(data)