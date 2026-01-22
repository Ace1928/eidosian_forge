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
class Test_icmpv6_router_solicit(unittest.TestCase):
    type_ = 133
    code = 0
    csum = 38873
    res = 0
    nd_type = 1
    nd_length = 1
    nd_hw_src = '12:2d:a5:6d:bc:0f'
    data = b'\x00\x00\x00\x00\x01\x01\x12-\xa5m\xbc\x0f'
    buf = b'\x85\x00\x97\xd9'
    src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
    dst_ipv6 = '3ffe:501:0:1001::2'

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        rs = icmpv6.nd_router_solicit(self.res)
        self.assertEqual(rs.res, self.res)
        self.assertEqual(rs.option, None)

    def _test_parser(self, data=None):
        buf = self.buf + (data or b'')
        msg, n, _ = icmpv6.icmpv6.parser(buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        if data is not None:
            self.assertEqual(msg.data.res, self.res)
        self.assertEqual(n, None)
        if data:
            rs = msg.data.option
            self.assertEqual(rs.length, self.nd_length)
            self.assertEqual(rs.hw_src, self.nd_hw_src)
            self.assertEqual(rs.data, None)

    def test_parser_without_data(self):
        self._test_parser()

    def test_parser_with_data(self):
        self._test_parser(self.data)

    def test_serialize_without_data(self):
        rs = icmpv6.nd_router_solicit(self.res)
        prev = ipv6(6, 0, 0, 8, 64, 255, self.src_ipv6, self.dst_ipv6)
        rs_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, rs)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        res = struct.unpack_from(rs._PACK_STR, buf, icmp._MIN_LEN)
        data = buf[icmp._MIN_LEN + rs._MIN_LEN:]
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, rs_csum)
        self.assertEqual(res[0], self.res)
        self.assertEqual(data, b'')

    def test_serialize_with_data(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        rs = icmpv6.nd_router_solicit(self.res, nd_opt)
        prev = ipv6(6, 0, 0, 16, 64, 255, self.src_ipv6, self.dst_ipv6)
        rs_csum = icmpv6_csum(prev, self.buf + self.data)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, rs)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        res = struct.unpack_from(rs._PACK_STR, buf, icmp._MIN_LEN)
        nd_type, nd_length, nd_hw_src = struct.unpack_from(nd_opt._PACK_STR, buf, icmp._MIN_LEN + rs._MIN_LEN)
        data = buf[icmp._MIN_LEN + rs._MIN_LEN + 8:]
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, rs_csum)
        self.assertEqual(res[0], self.res)
        self.assertEqual(nd_type, self.nd_type)
        self.assertEqual(nd_length, self.nd_length)
        self.assertEqual(nd_hw_src, addrconv.mac.text_to_bin(self.nd_hw_src))

    def test_to_string(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        rs = icmpv6.nd_router_solicit(self.res, nd_opt)
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, rs)
        nd_opt_values = {'length': self.nd_length, 'hw_src': self.nd_hw_src, 'data': None}
        _nd_opt_str = ','.join(['%s=%s' % (k, repr(nd_opt_values[k])) for k, v in inspect.getmembers(nd_opt) if k in nd_opt_values])
        nd_opt_str = '%s(%s)' % (icmpv6.nd_option_sla.__name__, _nd_opt_str)
        rs_values = {'res': repr(rs.res), 'option': nd_opt_str}
        _rs_str = ','.join(['%s=%s' % (k, rs_values[k]) for k, v in inspect.getmembers(rs) if k in rs_values])
        rs_str = '%s(%s)' % (icmpv6.nd_router_solicit.__name__, _rs_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': rs_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_ROUTER_SOLICIT, data=icmpv6.nd_router_solicit())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_ROUTER_SOLICIT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_router_solicit._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_ROUTER_SOLICIT, data=icmpv6.nd_router_solicit(option=icmpv6.nd_option_sla()))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_ROUTER_SOLICIT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_router_solicit._PACK_STR, bytes(buf[4:8]))
        self.assertEqual(res[0], 0)
        res = struct.unpack(icmpv6.nd_option_sla._PACK_STR, bytes(buf[8:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_SLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_sla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))

    def test_json(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        rs = icmpv6.nd_router_solicit(self.res, nd_opt)
        ic1 = icmpv6.icmpv6(self.type_, self.code, self.csum, rs)
        jsondict = ic1.to_jsondict()
        ic2 = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(ic1), str(ic2))