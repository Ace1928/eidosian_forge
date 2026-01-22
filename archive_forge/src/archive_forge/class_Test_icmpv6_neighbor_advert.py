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
class Test_icmpv6_neighbor_advert(Test_icmpv6_neighbor_solicit):

    def setUp(self):
        self.type_ = 136
        self.csum = 47290
        self.res = 7
        self.dst = '3ffe:507:0:1:260:97ff:fe07:69ea'
        self.nd_type = 2
        self.nd_length = 1
        self.nd_data = None
        self.nd_hw_src = '00:60:97:07:69:ea'
        self.data = b'\x02\x01\x00`\x97\x07i\xea'
        self.buf = b'\x88\x00\xb8\xba\xe0\x00\x00\x00' + b'?\xfe\x05\x07\x00\x00\x00\x01' + b'\x02`\x97\xff\xfe\x07i\xea'

    def test_serialize_with_data(self):
        nd_opt = icmpv6.nd_option_tla(self.nd_length, self.nd_hw_src)
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
        nd_opt = icmpv6.nd_option_tla(self.nd_length, self.nd_hw_src)
        nd = icmpv6.nd_neighbor(self.res, self.dst, nd_opt)
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, nd)
        nd_opt_values = {'length': self.nd_length, 'hw_src': self.nd_hw_src, 'data': None}
        _nd_opt_str = ','.join(['%s=%s' % (k, repr(nd_opt_values[k])) for k, v in inspect.getmembers(nd_opt) if k in nd_opt_values])
        nd_opt_str = '%s(%s)' % (icmpv6.nd_option_tla.__name__, _nd_opt_str)
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
        ic = icmpv6.icmpv6(type_=icmpv6.ND_NEIGHBOR_ADVERT, data=icmpv6.nd_neighbor())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_NEIGHBOR_ADVERT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_neighbor._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ND_NEIGHBOR_ADVERT, data=icmpv6.nd_neighbor(option=icmpv6.nd_option_tla()))
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ND_NEIGHBOR_ADVERT)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.nd_neighbor._PACK_STR, bytes(buf[4:24]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        res = struct.unpack(icmpv6.nd_option_tla._PACK_STR, bytes(buf[24:]))
        self.assertEqual(res[0], icmpv6.ND_OPTION_TLA)
        self.assertEqual(res[1], len(icmpv6.nd_option_tla()) // 8)
        self.assertEqual(res[2], addrconv.mac.text_to_bin('00:00:00:00:00:00'))