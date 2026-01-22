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
class TestOFPPortMod(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPPortMod
    """
    port_no = 1119692796
    hw_addr = 'e8:fe:5e:a9:68:6c'
    config = 2226555987
    mask = 1678244809
    advertise = 2025421682

    def test_init(self):
        c = OFPPortMod(_Datapath, self.port_no, self.hw_addr, self.config, self.mask, self.advertise)
        self.assertEqual(self.port_no, c.port_no)
        self.assertEqual(self.hw_addr, c.hw_addr)
        self.assertEqual(self.config, c.config)
        self.assertEqual(self.mask, c.mask)
        self.assertEqual(self.advertise, c.advertise)

    def _test_serialize(self, port_no, hw_addr, config, mask, advertise):
        c = OFPPortMod(_Datapath, port_no, hw_addr, config, mask, advertise)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_PORT_MOD, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_PORT_MOD_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(res[0], ofproto.OFP_VERSION)
        self.assertEqual(res[1], ofproto.OFPT_PORT_MOD)
        self.assertEqual(res[2], len(c.buf))
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], port_no)
        self.assertEqual(res[5], addrconv.mac.text_to_bin(hw_addr))
        self.assertEqual(res[6], config)
        self.assertEqual(res[7], mask)
        self.assertEqual(res[8], advertise)

    def test_serialize_mid(self):
        self._test_serialize(self.port_no, self.hw_addr, self.config, self.mask, self.advertise)

    def test_serialize_max(self):
        port_no = ofproto.OFPP_ANY
        hw_addr = 'ff:ff:ff:ff:ff:ff'
        config = 4294967295
        mask = 4294967295
        advertise = 4294967295
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_min(self):
        port_no = 0
        hw_addr = '00:00:00:00:00:00'
        config = 0
        mask = 0
        advertise = 0
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p1(self):
        port_no = ofproto.OFPP_MAX
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_PORT_DOWN
        mask = ofproto.OFPPC_PORT_DOWN
        advertise = ofproto.OFPPF_10MB_HD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p2(self):
        port_no = ofproto.OFPP_IN_PORT
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_RECV
        mask = ofproto.OFPPC_NO_RECV
        advertise = ofproto.OFPPF_10MB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p3(self):
        port_no = ofproto.OFPP_TABLE
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_FWD
        mask = ofproto.OFPPC_NO_FWD
        advertise = ofproto.OFPPF_100MB_HD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p4(self):
        port_no = ofproto.OFPP_NORMAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_100MB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p5(self):
        port_no = ofproto.OFPP_FLOOD
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_1GB_HD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p6(self):
        port_no = ofproto.OFPP_ALL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_1GB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p7(self):
        port_no = ofproto.OFPP_CONTROLLER
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_10GB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p8(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_40GB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p9(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_100GB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p10(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_1TB_FD
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p11(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_OTHER
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p12(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_COPPER
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p13(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_FIBER
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p14(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_AUTONEG
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p15(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_PAUSE
        self._test_serialize(port_no, hw_addr, config, mask, advertise)

    def test_serialize_p16(self):
        port_no = ofproto.OFPP_LOCAL
        hw_addr = self.hw_addr
        config = ofproto.OFPPC_NO_PACKET_IN
        mask = ofproto.OFPPC_NO_PACKET_IN
        advertise = ofproto.OFPPF_PAUSE_ASYM
        self._test_serialize(port_no, hw_addr, config, mask, advertise)