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
class TestOFPErrorExperimenterMsg(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPErrorExperimenterMsg
    """

    def test_init(self):
        c = OFPErrorExperimenterMsg(_Datapath)
        self.assertEqual(c.type, 65535)
        self.assertEqual(c.exp_type, None)
        self.assertEqual(c.experimenter, None)
        self.assertEqual(c.data, None)

    def _test_parser(self, exp_type, experimenter, data=None):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_ERROR
        msg_len = ofproto.OFP_ERROR_MSG_SIZE
        xid = 2495926989
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        type_ = 65535
        fmt = ofproto.OFP_ERROR_EXPERIMENTER_MSG_PACK_STR
        buf += pack(fmt, type_, exp_type, experimenter)
        if data is not None:
            buf += data
        res = OFPErrorMsg.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(res.version, version)
        self.assertEqual(res.msg_type, msg_type)
        self.assertEqual(res.msg_len, msg_len)
        self.assertEqual(res.xid, xid)
        self.assertEqual(res.type, type_)
        self.assertEqual(res.exp_type, exp_type)
        self.assertEqual(res.experimenter, experimenter)
        if data is not None:
            self.assertEqual(res.data, data)

    def test_parser_mid(self):
        exp_type = 32768
        experimenter = 2147483648
        data = b'Error Experimenter Message.'
        self._test_parser(exp_type, experimenter, data)

    def test_parser_max(self):
        exp_type = 65535
        experimenter = 4294967295
        data = b'Error Experimenter Message.'.ljust(65519)
        self._test_parser(exp_type, experimenter, data)

    def test_parser_min(self):
        exp_type = 0
        experimenter = 0
        self._test_parser(exp_type, experimenter)