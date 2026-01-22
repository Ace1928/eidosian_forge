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
class TestOFPExperimenter(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPExperimenter
    """
    c = OFPExperimenter(_Datapath)

    def _test_parser(self, xid, experimenter, exp_type):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_EXPERIMENTER
        msg_len = ofproto.OFP_EXPERIMENTER_HEADER_SIZE
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_EXPERIMENTER_HEADER_PACK_STR
        buf += pack(fmt, experimenter, exp_type)
        res = OFPExperimenter.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(version, res.version)
        self.assertEqual(msg_type, res.msg_type)
        self.assertEqual(msg_len, res.msg_len)
        self.assertEqual(xid, res.xid)
        self.assertEqual(experimenter, res.experimenter)
        self.assertEqual(exp_type, res.exp_type)

    def test_parser_mid(self):
        xid = 2495926989
        experimenter = 2147483648
        exp_type = 1
        self._test_parser(xid, experimenter, exp_type)

    def test_parser_max(self):
        xid = 4294967295
        experimenter = 4294967295
        exp_type = 65535
        self._test_parser(xid, experimenter, exp_type)

    def test_parser_min(self):
        xid = 0
        experimenter = 0
        exp_type = 0
        self._test_parser(xid, experimenter, exp_type)