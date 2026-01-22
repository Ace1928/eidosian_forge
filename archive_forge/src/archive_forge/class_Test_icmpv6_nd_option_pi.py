import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet import icmpv6
from os_ken.lib.packet.ipv6 import ipv6
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_icmpv6_nd_option_pi(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_default_args(self):
        pi = icmpv6.nd_option_pi()
        buf = pi.serialize()
        res = struct.unpack(icmpv6.nd_option_pi._PACK_STR, bytes(buf))
        self.assertEqual(res[0], icmpv6.ND_OPTION_PI)
        self.assertEqual(res[1], len(icmpv6.nd_option_pi()) // 8)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        self.assertEqual(res[7], addrconv.ipv6.text_to_bin('::'))
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_ROUTER_ADVERT, data=icmpv6.nd_router_advert(options=[icmpv6.nd_option_pi()]))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_ROUTER_ADVERT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_router_advert._PACK_STR, bytes(buf[4:16]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 0)
        res = struct.unpack(icmpv6.nd_option_pi._PACK_STR, bytes(buf[16:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_PI)
        self.assertEqual(res[1], 4)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        self.assertEqual(res[7], addrconv.ipv6.text_to_bin('::'))