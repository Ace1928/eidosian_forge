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
class Test_icmpv6_header(unittest.TestCase):
    type_ = 255
    code = 0
    csum = 207
    buf = b'\xff\x00\x00\xcf'
    icmp = icmpv6.icmpv6(type_, code, 0)

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.type_, self.icmp.type_)
        self.assertEqual(self.code, self.icmp.code)
        self.assertEqual(0, self.icmp.csum)

    def test_parser(self):
        msg, n, _ = self.icmp.parser(self.buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data, b'')
        self.assertEqual(n, None)

    def test_serialize(self):
        src_ipv6 = 'fe80::200:ff:fe00:ef'
        dst_ipv6 = 'fe80::200:ff:fe00:1'
        prev = ipv6(6, 0, 0, 4, 58, 255, src_ipv6, dst_ipv6)
        buf = self.icmp.serialize(bytearray(), prev)
        type_, code, csum = struct.unpack(self.icmp._PACK_STR, bytes(buf))
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, self.csum)

    def test_malformed_icmpv6(self):
        m_short_buf = self.buf[1:self.icmp._MIN_LEN]
        self.assertRaises(struct.error, self.icmp.parser, m_short_buf)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6()
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))

    def test_json(self):
        jsondict = self.icmp.to_jsondict()
        icmp = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(self.icmp), str(icmp))