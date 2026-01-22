import unittest
import inspect
import logging
from struct import pack, unpack_from, pack_into
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.packet_utils import checksum
from os_ken.lib import addrconv
from os_ken.lib.packet.igmp import igmp
from os_ken.lib.packet.igmp import igmpv3_query
from os_ken.lib.packet.igmp import igmpv3_report
from os_ken.lib.packet.igmp import igmpv3_report_group
from os_ken.lib.packet.igmp import IGMP_TYPE_QUERY
from os_ken.lib.packet.igmp import IGMP_TYPE_REPORT_V3
from os_ken.lib.packet.igmp import MODE_IS_INCLUDE
class Test_igmpv3_query(unittest.TestCase):
    """ Test case for Internet Group Management Protocol v3
    Membership Query Message"""

    def setUp(self):
        self.msgtype = IGMP_TYPE_QUERY
        self.maxresp = 100
        self.csum = 0
        self.address = '225.0.0.1'
        self.s_flg = 0
        self.qrv = 2
        self.qqic = 10
        self.num = 0
        self.srcs = []
        self.s_qrv = self.s_flg << 3 | self.qrv
        self.buf = pack(igmpv3_query._PACK_STR, self.msgtype, self.maxresp, self.csum, addrconv.ipv4.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        self.g = igmpv3_query(self.msgtype, self.maxresp, self.csum, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)

    def setUp_with_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs)
        self.buf = pack(igmpv3_query._PACK_STR, self.msgtype, self.maxresp, self.csum, addrconv.ipv4.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_query(self.msgtype, self.maxresp, self.csum, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)

    def tearDown(self):
        pass

    def find_protocol(self, pkt, name):
        for p in pkt.protocols:
            if p.protocol_name == name:
                return p

    def test_init(self):
        self.assertEqual(self.msgtype, self.g.msgtype)
        self.assertEqual(self.maxresp, self.g.maxresp)
        self.assertEqual(self.csum, self.g.csum)
        self.assertEqual(self.address, self.g.address)
        self.assertEqual(self.s_flg, self.g.s_flg)
        self.assertEqual(self.qrv, self.g.qrv)
        self.assertEqual(self.qqic, self.g.qqic)
        self.assertEqual(self.num, self.g.num)
        self.assertEqual(self.srcs, self.g.srcs)

    def test_init_with_srcs(self):
        self.setUp_with_srcs()
        self.test_init()

    def test_parser(self):
        _res = self.g.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(res.msgtype, self.msgtype)
        self.assertEqual(res.maxresp, self.maxresp)
        self.assertEqual(res.csum, self.csum)
        self.assertEqual(res.address, self.address)
        self.assertEqual(res.s_flg, self.s_flg)
        self.assertEqual(res.qrv, self.qrv)
        self.assertEqual(res.qqic, self.qqic)
        self.assertEqual(res.num, self.num)
        self.assertEqual(res.srcs, self.srcs)

    def test_parser_with_srcs(self):
        self.setUp_with_srcs()
        self.test_parser()

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.g.serialize(data, prev)
        res = unpack_from(igmpv3_query._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.msgtype)
        self.assertEqual(res[1], self.maxresp)
        self.assertEqual(res[2], checksum(self.buf))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
        self.assertEqual(res[4], self.s_qrv)
        self.assertEqual(res[5], self.qqic)
        self.assertEqual(res[6], self.num)

    def test_serialize_with_srcs(self):
        self.setUp_with_srcs()
        data = bytearray()
        prev = None
        buf = self.g.serialize(data, prev)
        res = unpack_from(igmpv3_query._PACK_STR, bytes(buf))
        src1, src2, src3 = unpack_from('4s4s4s', bytes(buf), igmpv3_query._MIN_LEN)
        self.assertEqual(res[0], self.msgtype)
        self.assertEqual(res[1], self.maxresp)
        self.assertEqual(res[2], checksum(self.buf))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
        self.assertEqual(res[4], self.s_qrv)
        self.assertEqual(res[5], self.qqic)
        self.assertEqual(res[6], self.num)
        self.assertEqual(src1, addrconv.ipv4.text_to_bin(self.srcs[0]))
        self.assertEqual(src2, addrconv.ipv4.text_to_bin(self.srcs[1]))
        self.assertEqual(src3, addrconv.ipv4.text_to_bin(self.srcs[2]))

    def _build_igmp(self):
        dl_dst = '11:22:33:44:55:66'
        dl_src = 'aa:bb:cc:dd:ee:ff'
        dl_type = ether.ETH_TYPE_IP
        e = ethernet(dl_dst, dl_src, dl_type)
        total_length = len(ipv4()) + len(self.g)
        nw_proto = inet.IPPROTO_IGMP
        nw_dst = '11.22.33.44'
        nw_src = '55.66.77.88'
        i = ipv4(total_length=total_length, src=nw_src, dst=nw_dst, proto=nw_proto, ttl=1)
        p = Packet()
        p.add_protocol(e)
        p.add_protocol(i)
        p.add_protocol(self.g)
        p.serialize()
        return p

    def test_build_igmp(self):
        p = self._build_igmp()
        e = self.find_protocol(p, 'ethernet')
        self.assertTrue(e)
        self.assertEqual(e.ethertype, ether.ETH_TYPE_IP)
        i = self.find_protocol(p, 'ipv4')
        self.assertTrue(i)
        self.assertEqual(i.proto, inet.IPPROTO_IGMP)
        g = self.find_protocol(p, 'igmpv3_query')
        self.assertTrue(g)
        self.assertEqual(g.msgtype, self.msgtype)
        self.assertEqual(g.maxresp, self.maxresp)
        self.assertEqual(g.csum, checksum(self.buf))
        self.assertEqual(g.address, self.address)
        self.assertEqual(g.s_flg, self.s_flg)
        self.assertEqual(g.qrv, self.qrv)
        self.assertEqual(g.qqic, self.qqic)
        self.assertEqual(g.num, self.num)
        self.assertEqual(g.srcs, self.srcs)

    def test_build_igmp_with_srcs(self):
        self.setUp_with_srcs()
        self.test_build_igmp()

    def test_to_string(self):
        igmp_values = {'msgtype': repr(self.msgtype), 'maxresp': repr(self.maxresp), 'csum': repr(self.csum), 'address': repr(self.address), 's_flg': repr(self.s_flg), 'qrv': repr(self.qrv), 'qqic': repr(self.qqic), 'num': repr(self.num), 'srcs': repr(self.srcs)}
        _g_str = ','.join(['%s=%s' % (k, igmp_values[k]) for k, v in inspect.getmembers(self.g) if k in igmp_values])
        g_str = '%s(%s)' % (igmpv3_query.__name__, _g_str)
        self.assertEqual(str(self.g), g_str)
        self.assertEqual(repr(self.g), g_str)

    def test_to_string_with_srcs(self):
        self.setUp_with_srcs()
        self.test_to_string()

    def test_num_larger_than_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs) + 1
        self.buf = pack(igmpv3_query._PACK_STR, self.msgtype, self.maxresp, self.csum, addrconv.ipv4.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_query(self.msgtype, self.maxresp, self.csum, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)
        self.assertRaises(Exception, self.test_parser)

    def test_num_smaller_than_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs) - 1
        self.buf = pack(igmpv3_query._PACK_STR, self.msgtype, self.maxresp, self.csum, addrconv.ipv4.text_to_bin(self.address), self.s_qrv, self.qqic, self.num)
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_query(self.msgtype, self.maxresp, self.csum, self.address, self.s_flg, self.qrv, self.qqic, self.num, self.srcs)
        self.assertRaises(Exception, self.test_parser)

    def test_default_args(self):
        prev = ipv4(proto=inet.IPPROTO_IGMP)
        g = igmpv3_query()
        prev.serialize(g, None)
        buf = g.serialize(bytearray(), prev)
        res = unpack_from(igmpv3_query._PACK_STR, bytes(buf))
        buf = bytearray(buf)
        pack_into('!H', buf, 2, 0)
        self.assertEqual(res[0], IGMP_TYPE_QUERY)
        self.assertEqual(res[1], 100)
        self.assertEqual(res[2], checksum(buf))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))
        self.assertEqual(res[4], 2)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        prev = ipv4(proto=inet.IPPROTO_IGMP)
        srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        g = igmpv3_query(srcs=srcs)
        prev.serialize(g, None)
        buf = g.serialize(bytearray(), prev)
        res = unpack_from(igmpv3_query._PACK_STR, bytes(buf))
        buf = bytearray(buf)
        pack_into('!H', buf, 2, 0)
        self.assertEqual(res[0], IGMP_TYPE_QUERY)
        self.assertEqual(res[1], 100)
        self.assertEqual(res[2], checksum(buf))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))
        self.assertEqual(res[4], 2)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], len(srcs))
        res = unpack_from('4s4s4s', bytes(buf), igmpv3_query._MIN_LEN)
        self.assertEqual(res[0], addrconv.ipv4.text_to_bin(srcs[0]))
        self.assertEqual(res[1], addrconv.ipv4.text_to_bin(srcs[1]))
        self.assertEqual(res[2], addrconv.ipv4.text_to_bin(srcs[2]))

    def test_json(self):
        jsondict = self.g.to_jsondict()
        g = igmpv3_query.from_jsondict(jsondict['igmpv3_query'])
        self.assertEqual(str(self.g), str(g))

    def test_json_with_srcs(self):
        self.setUp_with_srcs()
        self.test_json()