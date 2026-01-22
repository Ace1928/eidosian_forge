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
class TestOFPSwitchFeatures(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPSwitchFeatures
    """

    def _test_parser(self, xid, datapath_id, n_buffers, n_tables, capabilities, reserved, port_cnt=0):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_FEATURES_REPLY
        msg_len = ofproto.OFP_SWITCH_FEATURES_SIZE + ofproto.OFP_PORT_SIZE * port_cnt
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        fmt = ofproto.OFP_SWITCH_FEATURES_PACK_STR
        buf += pack(fmt, datapath_id, n_buffers, n_tables, capabilities, reserved)
        for i in range(port_cnt):
            port_no = i
            fmt = ofproto.OFP_PORT_PACK_STR
            buf += pack(fmt, port_no, b'\x00' * 6, b'\x00' * 16, 0, 0, 0, 0, 0, 0, 0, 0)
        res = OFPSwitchFeatures.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(res.version, version)
        self.assertEqual(res.msg_type, msg_type)
        self.assertEqual(res.msg_len, msg_len)
        self.assertEqual(res.xid, xid)
        self.assertEqual(res.datapath_id, datapath_id)
        self.assertEqual(res.n_buffers, n_buffers)
        self.assertEqual(res.n_tables, n_tables)
        self.assertEqual(res.capabilities, capabilities)
        self.assertEqual(res._reserved, reserved)
        for i in range(port_cnt):
            self.assertEqual(res.ports[i].port_no, i)

    def test_parser_mid(self):
        xid = 2495926989
        datapath_id = 1270985291017894273
        n_buffers = 2148849654
        n_tables = 228
        capabilities = 1766843586
        reserved = 2013714700
        port_cnt = 1
        self._test_parser(xid, datapath_id, n_buffers, n_tables, capabilities, reserved, port_cnt)

    def test_parser_max(self):
        xid = 4294967295
        datapath_id = 18446744073709551615
        n_buffers = 4294967295
        n_tables = 255
        capabilities = 4294967295
        reserved = 4294967295
        port_cnt = 1023
        self._test_parser(xid, datapath_id, n_buffers, n_tables, capabilities, reserved, port_cnt)

    def test_parser_min(self):
        xid = 0
        datapath_id = 0
        n_buffers = 0
        n_tables = 0
        capabilities = 0
        reserved = 0
        port_cnt = 0
        self._test_parser(xid, datapath_id, n_buffers, n_tables, capabilities, reserved, port_cnt)