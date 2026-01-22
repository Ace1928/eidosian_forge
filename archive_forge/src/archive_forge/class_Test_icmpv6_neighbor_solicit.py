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
class Test_icmpv6_neighbor_solicit(unittest.TestCase):
    type_ = 135
    code = 0
    csum = 38189
    res = 0
    dst = '3ffe:507:0:1:200:86ff:fe05:80da'
    nd_type = 1
    nd_length = 1
    nd_hw_src = '00:60:97:07:69:ea'
    data = b'\x01\x01\x00`\x97\x07i\xea'
    buf = b'\x87\x00\x95-\x00\x00\x00\x00' + b'?\xfe\x05\x07\x00\x00\x00\x01' + b'\x02\x00\x86\xff\xfe\x05\x80\xda'
    src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
    dst_ipv6 = '3ffe:501:0:1001::2'

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        nd = icmpv6.nd_neighbor(self.res, self.dst)
        self.assertEqual(nd.res, self.res)
        self.assertEqual(nd.dst, self.dst)
        self.assertEqual(nd.option, None)

    def _test_parser(self, data=None):
        buf = self.buf + (data or b'')
        msg, n, _ = icmpv6.icmpv6.parser(buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data.res, self.res)
        self.assertEqual(addrconv.ipv6.text_to_bin(msg.data.dst), addrconv.ipv6.text_to_bin(self.dst))
        self.assertEqual(n, None)
        if data:
            nd = msg.data.option
            self.assertEqual(nd.length, self.nd_length)
            self.assertEqual(nd.hw_src, self.nd_hw_src)
            self.assertEqual(nd.data, None)

    def test_parser_without_data(self):
        self._test_parser()

    def test_parser_with_data(self):
        self._test_parser(self.data)

    def test_serialize_without_data(self):
        nd = icmpv6.nd_neighbor(self.res, self.dst)
        prev = ipv6(6, 0, 0, 24, 64, 255, self.src_ipv6, self.dst_ipv6)
        nd_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, nd)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        res, dst = struct.unpack_from(nd._PACK_STR, buf, icmp._MIN_LEN)
        data = buf[icmp._MIN_LEN + nd._MIN_LEN:]
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, nd_csum)
        self.assertEqual(res >> 29, self.res)
        self.assertEqual(dst, addrconv.ipv6.text_to_bin(self.dst))
        self.assertEqual(data, b'')

    def test_serialize_with_data(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        nd = icmpv6.nd_neighbor(self.res, self.dst, nd_opt)
        prev = ipv6(6, 0, 0, 32, 64, 255, self.src_ipv6, self.dst_ipv6)
        nd_csum = icmpv6_csum(prev, self.buf + self.data)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, nd)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        res, dst = struct.unpack_from(nd._PACK_STR, buf, icmp._MIN_LEN)
        nd_type, nd_length, nd_hw_src = struct.unpack_from(nd_opt._PACK_STR, buf, icmp._MIN_LEN + nd._MIN_LEN)
        data = buf[icmp._MIN_LEN + nd._MIN_LEN + 8:]
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, nd_csum)
        self.assertEqual(res >> 29, self.res)
        self.assertEqual(dst, addrconv.ipv6.text_to_bin(self.dst))
        self.assertEqual(nd_type, self.nd_type)
        self.assertEqual(nd_length, self.nd_length)
        self.assertEqual(nd_hw_src, addrconv.mac.text_to_bin(self.nd_hw_src))

    def test_to_string(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        nd = icmpv6.nd_neighbor(self.res, self.dst, nd_opt)
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, nd)
        nd_opt_values = {'length': self.nd_length, 'hw_src': self.nd_hw_src, 'data': None}
        _nd_opt_str = ','.join(['%s=%s' % (k, repr(nd_opt_values[k])) for k, v in inspect.getmembers(nd_opt) if k in nd_opt_values])
        nd_opt_str = '%s(%s)' % (icmpv6.nd_option_sla.__name__, _nd_opt_str)
        nd_values = {'res': repr(nd.res), 'dst': repr(self.dst), 'option': nd_opt_str}
        _nd_str = ','.join(['%s=%s' % (k, nd_values[k]) for k, v in inspect.getmembers(nd) if k in nd_values])
        nd_str = '%s(%s)' % (icmpv6.nd_neighbor.__name__, _nd_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': nd_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_NEIGHBOR_SOLICIT, data=icmpv6.nd_neighbor())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_NEIGHBOR_SOLICIT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_neighbor._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_NEIGHBOR_SOLICIT, data=icmpv6.nd_neighbor(option=icmpv6.nd_option_sla()))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_NEIGHBOR_SOLICIT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_neighbor._PACK_STR, bytes(buf[4:24]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        res = struct.unpack(icmpv6.nd_option_sla._PACK_STR, bytes(buf[24:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_SLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_sla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))

    def test_json(self):
        nd_opt = icmpv6.nd_option_sla(self.nd_length, self.nd_hw_src)
        nd = icmpv6.nd_neighbor(self.res, self.dst, nd_opt)
        ic1 = icmpv6.icmpv6(self.type_, self.code, self.csum, nd)
        jsondict = ic1.to_jsondict()
        ic2 = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(ic1), str(ic2))