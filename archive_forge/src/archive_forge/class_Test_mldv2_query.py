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
class Test_mldv2_query(unittest.TestCase):
    type_ = 130
    code = 0
    csum = 46500
    maxresp = 10000
    address = 'ff08::1'
    s_flg = 0
    qrv = 2
    s_qrv = s_flg << 3 | qrv
    qqic = 10
    num = 0
    srcs = []
    mld = icmpv6.mldv2_query(maxresp, address, s_flg, qrv, qqic, num, srcs)
    buf = b"\x82\x00\xb5\xa4'\x10\x00\x00" + b'\xff\x08\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x02\n\x00\x00'

    def setUp(self):
        pass

    def setUp_with_srcs(self):
        self.num = 2
        self.srcs = ['ff80::1', 'ff80::2']
        self.mld = icmpv6.mldv2_query(self.maxresp, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)
        self.buf = b"\x82\x00\xb5\xa4'\x10\x00\x00" + b'\xff\x08\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\x02\n\x00\x02' + b'\xff\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x01' + b'\xff\x80\x00\x00\x00\x00\x00\x00' + b'\x00\x00\x00\x00\x00\x00\x00\x02'

    def tearDown(self):
        pass

    def find_protocol(self, pkt, name):
        for p in pkt.protocols:
            if p.protocol_name == name:
                return p

    def test_init(self):
        self.assertEqual(self.mld.maxresp, self.maxresp)
        self.assertEqual(self.mld.address, self.address)
        self.assertEqual(self.mld.s_flg, self.s_flg)
        self.assertEqual(self.mld.qrv, self.qrv)
        self.assertEqual(self.mld.qqic, self.qqic)
        self.assertEqual(self.mld.num, self.num)
        self.assertEqual(self.mld.srcs, self.srcs)

    def test_init_with_srcs(self):
        self.setUp_with_srcs()
        self.test_init()

    def test_parser(self):
        msg, n, _ = icmpv6.icmpv6.parser(self.buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data.maxresp, self.maxresp)
        self.assertEqual(msg.data.address, self.address)
        self.assertEqual(msg.data.s_flg, self.s_flg)
        self.assertEqual(msg.data.qrv, self.qrv)
        self.assertEqual(msg.data.qqic, self.qqic)
        self.assertEqual(msg.data.num, self.num)
        self.assertEqual(msg.data.srcs, self.srcs)
        self.assertEqual(n, None)

    def test_parser_with_srcs(self):
        self.setUp_with_srcs()
        self.test_parser()

    def test_serialize(self):
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(self.buf), 64, 255, src_ipv6, dst_ipv6)
        mld_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, self.mld)
        buf = icmp.serialize(bytearray(), prev)
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, bytes(buf))
        maxresp, address, s_qrv, qqic, num = struct.unpack_from(self.mld._PACK_STR, bytes(buf), icmp._MIN_LEN)
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, mld_csum)
        self.assertEqual(maxresp, self.maxresp)
        self.assertEqual(address, addrconv.ipv6.text_to_bin(self.address))
        s_flg = s_qrv >> 3 & 1
        qrv = s_qrv & 7
        self.assertEqual(s_flg, self.s_flg)
        self.assertEqual(qrv, self.qrv)
        self.assertEqual(qqic, self.qqic)
        self.assertEqual(num, self.num)

    def test_serialize_with_srcs(self):
        self.setUp_with_srcs()
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(self.buf), 64, 255, src_ipv6, dst_ipv6)
        mld_csum = icmpv6_csum(prev, self.buf)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, self.mld)
        buf = icmp.serialize(bytearray(), prev)
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, bytes(buf))
        maxresp, address, s_qrv, qqic, num = struct.unpack_from(self.mld._PACK_STR, bytes(buf), icmp._MIN_LEN)
        addr1, addr2 = struct.unpack_from('!16s16s', bytes(buf), icmp._MIN_LEN + self.mld._MIN_LEN)
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, mld_csum)
        self.assertEqual(maxresp, self.maxresp)
        self.assertEqual(address, addrconv.ipv6.text_to_bin(self.address))
        s_flg = s_qrv >> 3 & 1
        qrv = s_qrv & 7
        self.assertEqual(s_flg, self.s_flg)
        self.assertEqual(qrv, self.qrv)
        self.assertEqual(qqic, self.qqic)
        self.assertEqual(num, self.num)
        self.assertEqual(addr1, addrconv.ipv6.text_to_bin(self.srcs[0]))
        self.assertEqual(addr2, addrconv.ipv6.text_to_bin(self.srcs[1]))

    def _build_mldv2_query(self):
        e = ethernet(ethertype=ether.ETH_TYPE_IPV6)
        i = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_QUERY, data=self.mld)
        p = e / i / ic
        return p

    def test_build_mldv2_query(self):
        p = self._build_mldv2_query()
        e = self.find_protocol(p, 'ethernet')
        self.assertTrue(e)
        self.assertEqual(e.ethertype, ether.ETH_TYPE_IPV6)
        i = self.find_protocol(p, 'ipv6')
        self.assertTrue(i)
        self.assertEqual(i.nxt, inet.IPPROTO_ICMPV6)
        ic = self.find_protocol(p, 'icmpv6')
        self.assertTrue(ic)
        self.assertEqual(ic.type_, icmpv6.MLD_LISTENER_QUERY)
        self.assertEqual(ic.data.maxresp, self.maxresp)
        self.assertEqual(ic.data.address, self.address)
        self.assertEqual(ic.data.s_flg, self.s_flg)
        self.assertEqual(ic.data.qrv, self.qrv)
        self.assertEqual(ic.data.num, self.num)
        self.assertEqual(ic.data.srcs, self.srcs)

    def test_build_mldv2_query_with_srcs(self):
        self.setUp_with_srcs()
        self.test_build_mldv2_query()

    def test_to_string(self):
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, self.mld)
        mld_values = {'maxresp': self.maxresp, 'address': self.address, 's_flg': self.s_flg, 'qrv': self.qrv, 'qqic': self.qqic, 'num': self.num, 'srcs': self.srcs}
        _mld_str = ','.join(['%s=%s' % (k, repr(mld_values[k])) for k, v in inspect.getmembers(self.mld) if k in mld_values])
        mld_str = '%s(%s)' % (icmpv6.mldv2_query.__name__, _mld_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': mld_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_to_string_with_srcs(self):
        self.setUp_with_srcs()
        self.test_to_string()

    def test_num_larger_than_srcs(self):
        self.srcs = ['ff80::1', 'ff80::2', 'ff80::3']
        self.num = len(self.srcs) + 1
        self.buf = struct.pack(icmpv6.mldv2_query._PACK_STR, self.maxresp, addrconv.ipv6.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        for src in self.srcs:
            self.buf += struct.pack('16s', addrconv.ipv6.text_to_bin(src))
        self.mld = icmpv6.mldv2_query(self.maxresp, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)
        self.assertRaises(AssertionError, self.test_parser)

    def test_num_smaller_than_srcs(self):
        self.srcs = ['ff80::1', 'ff80::2', 'ff80::3']
        self.num = len(self.srcs) - 1
        self.buf = struct.pack(icmpv6.mldv2_query._PACK_STR, self.maxresp, addrconv.ipv6.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        for src in self.srcs:
            self.buf += struct.pack('16s', addrconv.ipv6.text_to_bin(src))
        self.mld = icmpv6.mldv2_query(self.maxresp, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)
        self.assertRaises(AssertionError, self.test_parser)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.MLD_LISTENER_QUERY, data=icmpv6.mldv2_query())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.MLD_LISTENER_QUERY)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.mldv2_query._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        self.assertEqual(res[2], 2)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 0)
        srcs = ['ff80::1', 'ff80::2', 'ff80::3']
        que = icmpv6.mldv2_query(srcs=srcs)
        buf = que.serialize()
        res = struct.unpack_from(icmpv6.mldv2_query._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], addrconv.ipv6.text_to_bin('::'))
        self.assertEqual(res[2], 2)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], len(srcs))
        src1, src2, src3 = struct.unpack_from('16s16s16s', bytes(buf), icmpv6.mldv2_query._MIN_LEN)
        self.assertEqual(src1, addrconv.ipv6.text_to_bin(srcs[0]))
        self.assertEqual(src2, addrconv.ipv6.text_to_bin(srcs[1]))
        self.assertEqual(src3, addrconv.ipv6.text_to_bin(srcs[2]))

    def test_json(self):
        jsondict = self.mld.to_jsondict()
        mld = icmpv6.mldv2_query.from_jsondict(jsondict['mldv2_query'])
        self.assertEqual(str(self.mld), str(mld))

    def test_json_with_srcs(self):
        self.setUp_with_srcs()
        self.test_json()