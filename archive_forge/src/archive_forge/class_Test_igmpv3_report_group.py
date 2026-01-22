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
class Test_igmpv3_report_group(unittest.TestCase):
    """Test case for Group Records of
    Internet Group Management Protocol v3 Membership Report Message"""

    def setUp(self):
        self.type_ = MODE_IS_INCLUDE
        self.aux_len = 0
        self.num = 0
        self.address = '225.0.0.1'
        self.srcs = []
        self.aux = None
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)

    def setUp_with_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs)
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)

    def setUp_with_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x00\x00\x00'
        self.aux_len = len(self.aux) // 4
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        self.buf += self.aux
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)

    def setUp_with_srcs_and_aux(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs)
        self.aux = b'\x01\x02\x03\x04\x05\x00\x00\x00'
        self.aux_len = len(self.aux) // 4
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.buf += self.aux
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.type_, self.g.type_)
        self.assertEqual(self.aux_len, self.g.aux_len)
        self.assertEqual(self.num, self.g.num)
        self.assertEqual(self.address, self.g.address)
        self.assertEqual(self.srcs, self.g.srcs)
        self.assertEqual(self.aux, self.g.aux)

    def test_init_with_srcs(self):
        self.setUp_with_srcs()
        self.test_init()

    def test_init_with_aux(self):
        self.setUp_with_aux()
        self.test_init()

    def test_init_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_init()

    def test_parser(self):
        _res = self.g.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(res.type_, self.type_)
        self.assertEqual(res.aux_len, self.aux_len)
        self.assertEqual(res.num, self.num)
        self.assertEqual(res.address, self.address)
        self.assertEqual(res.srcs, self.srcs)
        self.assertEqual(res.aux, self.aux)

    def test_parser_with_srcs(self):
        self.setUp_with_srcs()
        self.test_parser()

    def test_parser_with_aux(self):
        self.setUp_with_aux()
        self.test_parser()

    def test_parser_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_parser()

    def test_serialize(self):
        buf = self.g.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))

    def test_serialize_with_srcs(self):
        self.setUp_with_srcs()
        buf = self.g.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        src1, src2, src3 = unpack_from('4s4s4s', bytes(buf), igmpv3_report_group._MIN_LEN)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
        self.assertEqual(src1, addrconv.ipv4.text_to_bin(self.srcs[0]))
        self.assertEqual(src2, addrconv.ipv4.text_to_bin(self.srcs[1]))
        self.assertEqual(src3, addrconv.ipv4.text_to_bin(self.srcs[2]))

    def test_serialize_with_aux(self):
        self.setUp_with_aux()
        buf = self.g.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        aux, = unpack_from('%ds' % (self.aux_len * 4), bytes(buf), igmpv3_report_group._MIN_LEN)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
        self.assertEqual(aux, self.aux)

    def test_serialize_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        buf = self.g.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        src1, src2, src3 = unpack_from('4s4s4s', bytes(buf), igmpv3_report_group._MIN_LEN)
        aux, = unpack_from('%ds' % (self.aux_len * 4), bytes(buf), igmpv3_report_group._MIN_LEN + 12)
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.aux_len)
        self.assertEqual(res[2], self.num)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin(self.address))
        self.assertEqual(src1, addrconv.ipv4.text_to_bin(self.srcs[0]))
        self.assertEqual(src2, addrconv.ipv4.text_to_bin(self.srcs[1]))
        self.assertEqual(src3, addrconv.ipv4.text_to_bin(self.srcs[2]))
        self.assertEqual(aux, self.aux)

    def test_to_string(self):
        igmp_values = {'type_': repr(self.type_), 'aux_len': repr(self.aux_len), 'num': repr(self.num), 'address': repr(self.address), 'srcs': repr(self.srcs), 'aux': repr(self.aux)}
        _g_str = ','.join(['%s=%s' % (k, igmp_values[k]) for k, v in inspect.getmembers(self.g) if k in igmp_values])
        g_str = '%s(%s)' % (igmpv3_report_group.__name__, _g_str)
        self.assertEqual(str(self.g), g_str)
        self.assertEqual(repr(self.g), g_str)

    def test_to_string_with_srcs(self):
        self.setUp_with_srcs()
        self.test_to_string()

    def test_to_string_with_aux(self):
        self.setUp_with_aux()
        self.test_to_string()

    def test_to_string_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.test_to_string()

    def test_len(self):
        self.assertEqual(len(self.g), 8)

    def test_len_with_srcs(self):
        self.setUp_with_srcs()
        self.assertEqual(len(self.g), 20)

    def test_len_with_aux(self):
        self.setUp_with_aux()
        self.assertEqual(len(self.g), 16)

    def test_len_with_srcs_and_aux(self):
        self.setUp_with_srcs_and_aux()
        self.assertEqual(len(self.g), 28)

    def test_num_larger_than_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs) + 1
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_num_smaller_than_srcs(self):
        self.srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        self.num = len(self.srcs) - 1
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        for src in self.srcs:
            self.buf += pack('4s', addrconv.ipv4.text_to_bin(src))
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_aux_len_larger_than_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x00\x00\x00'
        self.aux_len = len(self.aux) // 4 + 1
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        self.buf += self.aux
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(Exception, self.test_parser)

    def test_aux_len_smaller_than_aux(self):
        self.aux = b'\x01\x02\x03\x04\x05\x00\x00\x00'
        self.aux_len = len(self.aux) // 4 - 1
        self.buf = pack(igmpv3_report_group._PACK_STR, self.type_, self.aux_len, self.num, addrconv.ipv4.text_to_bin(self.address))
        self.buf += self.aux
        self.g = igmpv3_report_group(self.type_, self.aux_len, self.num, self.address, self.srcs, self.aux)
        self.assertRaises(AssertionError, self.test_parser)

    def test_default_args(self):
        rep = igmpv3_report_group()
        buf = rep.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))
        srcs = ['192.168.1.1', '192.168.1.2', '192.168.1.3']
        rep = igmpv3_report_group(srcs=srcs)
        buf = rep.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], len(srcs))
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))
        res = unpack_from('4s4s4s', bytes(buf), igmpv3_report_group._MIN_LEN)
        self.assertEqual(res[0], addrconv.ipv4.text_to_bin(srcs[0]))
        self.assertEqual(res[1], addrconv.ipv4.text_to_bin(srcs[1]))
        self.assertEqual(res[2], addrconv.ipv4.text_to_bin(srcs[2]))
        aux = b'abcde'
        rep = igmpv3_report_group(aux=aux)
        buf = rep.serialize()
        res = unpack_from(igmpv3_report_group._PACK_STR, bytes(buf))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 2)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], addrconv.ipv4.text_to_bin('0.0.0.0'))
        self.assertEqual(buf[igmpv3_report_group._MIN_LEN:], b'abcde\x00\x00\x00')