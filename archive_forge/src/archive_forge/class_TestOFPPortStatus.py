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
class TestOFPPortStatus(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPortStatus
    """

    def _test_parser(self, xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed):
        version = ofproto.OFP_VERSION
        msg_type = ofproto.OFPT_PORT_STATUS
        msg_len = ofproto.OFP_PORT_STATUS_SIZE
        fmt = ofproto.OFP_HEADER_PACK_STR
        buf = pack(fmt, version, msg_type, msg_len, xid)
        hw_addr = '80:ff:9a:e3:72:85'
        name = b'name'.ljust(16)
        fmt = ofproto.OFP_PORT_STATUS_PACK_STR
        buf += pack(fmt, reason, port_no, addrconv.mac.text_to_bin(hw_addr), name, config, state, curr, advertised, supported, peer, curr_speed, max_speed)
        res = OFPPortStatus.parser(object, version, msg_type, msg_len, xid, buf)
        self.assertEqual(reason, res.reason)
        self.assertEqual(port_no, res.desc.port_no)
        self.assertEqual(hw_addr, res.desc.hw_addr)
        self.assertEqual(name, res.desc.name)
        self.assertEqual(config, res.desc.config)
        self.assertEqual(state, res.desc.state)
        self.assertEqual(curr, res.desc.curr)
        self.assertEqual(advertised, res.desc.advertised)
        self.assertEqual(supported, res.desc.supported)
        self.assertEqual(peer, res.desc.peer)
        self.assertEqual(curr_speed, res.desc.curr_speed)
        self.assertEqual(max_speed, res.desc.max_speed)

    def test_parser_mid(self):
        xid = 3423224276
        reason = 128
        port_no = 1119692796
        config = 2226555987
        state = 1678244809
        curr = 2850556459
        advertised = 2025421682
        supported = 2120575149
        peer = 2757463021
        curr_speed = 2641353507
        max_speed = 1797291672
        self._test_parser(xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_max(self):
        xid = 4294967295
        reason = 255
        port_no = ofproto.OFPP_ANY
        config = 4294967295
        state = 4294967295
        curr = 4294967295
        advertised = 4294967295
        supported = 4294967295
        peer = 4294967295
        curr_speed = 4294967295
        max_speed = 4294967295
        self._test_parser(xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_min(self):
        xid = 0
        reason = 0
        port_no = 0
        config = 0
        state = 0
        curr = 0
        advertised = 0
        supported = 0
        peer = 0
        curr_speed = 0
        max_speed = 0
        self._test_parser(xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p1(self):
        xid = 3423224276
        reason = ofproto.OFPPR_DELETE
        port_no = ofproto.OFPP_MAX
        config = ofproto.OFPPC_PORT_DOWN
        state = ofproto.OFPPS_LINK_DOWN
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_10MB_HD
        self._test_parser(xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed)

    def test_parser_p2(self):
        xid = 3423224276
        reason = ofproto.OFPPR_MODIFY
        port_no = ofproto.OFPP_MAX
        config = ofproto.OFPPC_PORT_DOWN
        state = ofproto.OFPPS_LINK_DOWN
        curr = advertised = supported = peer = curr_speed = max_speed = ofproto.OFPPF_10MB_HD
        self._test_parser(xid, reason, port_no, config, state, curr, advertised, supported, peer, curr_speed, max_speed)