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
class Test_igmpv3_report(unittest.TestCase):
    """ Test case for Internet Group Management Protocol v3
    Membership Report Message"""

    def setUp(self):
        self.msgtype = IGMP_TYPE_REPORT_V3
        self.csum = 0
        self.record_num = 0
        self.records = []
        self.buf = pack(igmpv3_report._PACK_STR, self.msgtype, self.csum, self.record_num)
        self.g = igmpv3_report(self.msgtype, self.csum, self.record_num, self.records)

    def setUp_with_records(self):
        self.record1 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 0, '225.0.0.1')
        self.record2 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 2, '225.0.0.2', ['172.16.10.10', '172.16.10.27'])
        self.record3 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 0, '225.0.0.3', [], b'abc\x00')
        self.record4 = igmpv3_report_group(MODE_IS_INCLUDE, 2, 2, '225.0.0.4', ['172.16.10.10', '172.16.10.27'], b'abcde\x00\x00\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records)
        self.buf = pack(igmpv3_report._PACK_STR, self.msgtype, self.csum, self.record_num)
        self.buf += self.record1.serialize()
        self.buf += self.record2.serialize()
        self.buf += self.record3.serialize()
        self.buf += self.record4.serialize()
        self.g = igmpv3_report(self.msgtype, self.csum, self.record_num, self.records)

    def tearDown(self):
        pass

    def find_protocol(self, pkt, name):
        for p in pkt.protocols:
            if p.protocol_name == name:
                return p

    def test_init(self):
        self.assertEqual(self.msgtype, self.g.msgtype)
        self.assertEqual(self.csum, self.g.csum)
        self.assertEqual(self.record_num, self.g.record_num)
        self.assertEqual(self.records, self.g.records)

    def test_init_with_records(self):
        self.setUp_with_records()
        self.test_init()

    def test_parser(self):
        _res = self.g.parser(bytes(self.buf))
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(res.msgtype, self.msgtype)
        self.assertEqual(res.csum, self.csum)
        self.assertEqual(res.record_num, self.record_num)
        self.assertEqual(repr(res.records), repr(self.records))

    def test_parser_with_records(self):
        self.setUp_with_records()
        self.test_parser()

    def test_serialize(self):
        data = bytearray()
        prev = None
        buf = self.g.serialize(data, prev)
        res = unpack_from(igmpv3_report._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.msgtype)
        self.assertEqual(res[1], checksum(self.buf))
        self.assertEqual(res[2], self.record_num)

    def test_serialize_with_records(self):
        self.setUp_with_records()
        data = bytearray()
        prev = None
        buf = bytes(self.g.serialize(data, prev))
        res = unpack_from(igmpv3_report._PACK_STR, buf)
        offset = igmpv3_report._MIN_LEN
        rec1 = igmpv3_report_group.parser(buf[offset:])
        offset += len(rec1)
        rec2 = igmpv3_report_group.parser(buf[offset:])
        offset += len(rec2)
        rec3 = igmpv3_report_group.parser(buf[offset:])
        offset += len(rec3)
        rec4 = igmpv3_report_group.parser(buf[offset:])
        self.assertEqual(res[0], self.msgtype)
        self.assertEqual(res[1], checksum(self.buf))
        self.assertEqual(res[2], self.record_num)
        self.assertEqual(repr(rec1), repr(self.record1))
        self.assertEqual(repr(rec2), repr(self.record2))
        self.assertEqual(repr(rec3), repr(self.record3))
        self.assertEqual(repr(rec4), repr(self.record4))

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
        g = self.find_protocol(p, 'igmpv3_report')
        self.assertTrue(g)
        self.assertEqual(g.msgtype, self.msgtype)
        self.assertEqual(g.csum, checksum(self.buf))
        self.assertEqual(g.record_num, self.record_num)
        self.assertEqual(g.records, self.records)

    def test_build_igmp_with_records(self):
        self.setUp_with_records()
        self.test_build_igmp()

    def test_to_string(self):
        igmp_values = {'msgtype': repr(self.msgtype), 'csum': repr(self.csum), 'record_num': repr(self.record_num), 'records': repr(self.records)}
        _g_str = ','.join(['%s=%s' % (k, igmp_values[k]) for k, v in inspect.getmembers(self.g) if k in igmp_values])
        g_str = '%s(%s)' % (igmpv3_report.__name__, _g_str)
        self.assertEqual(str(self.g), g_str)
        self.assertEqual(repr(self.g), g_str)

    def test_to_string_with_records(self):
        self.setUp_with_records()
        self.test_to_string()

    def test_record_num_larger_than_records(self):
        self.record1 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 0, '225.0.0.1')
        self.record2 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 2, '225.0.0.2', ['172.16.10.10', '172.16.10.27'])
        self.record3 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 0, '225.0.0.3', [], b'abc\x00')
        self.record4 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 2, '225.0.0.4', ['172.16.10.10', '172.16.10.27'], b'abc\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records) + 1
        self.buf = pack(igmpv3_report._PACK_STR, self.msgtype, self.csum, self.record_num)
        self.buf += self.record1.serialize()
        self.buf += self.record2.serialize()
        self.buf += self.record3.serialize()
        self.buf += self.record4.serialize()
        self.g = igmpv3_report(self.msgtype, self.csum, self.record_num, self.records)
        self.assertRaises(Exception, self.test_parser)

    def test_record_num_smaller_than_records(self):
        self.record1 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 0, '225.0.0.1')
        self.record2 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 2, '225.0.0.2', ['172.16.10.10', '172.16.10.27'])
        self.record3 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 0, '225.0.0.3', [], b'abc\x00')
        self.record4 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 2, '225.0.0.4', ['172.16.10.10', '172.16.10.27'], b'abc\x00')
        self.records = [self.record1, self.record2, self.record3, self.record4]
        self.record_num = len(self.records) - 1
        self.buf = pack(igmpv3_report._PACK_STR, self.msgtype, self.csum, self.record_num)
        self.buf += self.record1.serialize()
        self.buf += self.record2.serialize()
        self.buf += self.record3.serialize()
        self.buf += self.record4.serialize()
        self.g = igmpv3_report(self.msgtype, self.csum, self.record_num, self.records)
        self.assertRaises(Exception, self.test_parser)

    def test_default_args(self):
        prev = ipv4(proto=inet.IPPROTO_IGMP)
        g = igmpv3_report()
        prev.serialize(g, None)
        buf = g.serialize(bytearray(), prev)
        res = unpack_from(igmpv3_report._PACK_STR, bytes(buf))
        buf = bytearray(buf)
        pack_into('!H', buf, 2, 0)
        self.assertEqual(res[0], IGMP_TYPE_REPORT_V3)
        self.assertEqual(res[1], checksum(buf))
        self.assertEqual(res[2], 0)
        prev = ipv4(proto=inet.IPPROTO_IGMP)
        record1 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 0, '225.0.0.1')
        record2 = igmpv3_report_group(MODE_IS_INCLUDE, 0, 2, '225.0.0.2', ['172.16.10.10', '172.16.10.27'])
        record3 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 0, '225.0.0.3', [], b'abc\x00')
        record4 = igmpv3_report_group(MODE_IS_INCLUDE, 1, 2, '225.0.0.4', ['172.16.10.10', '172.16.10.27'], b'abc\x00')
        records = [record1, record2, record3, record4]
        g = igmpv3_report(records=records)
        prev.serialize(g, None)
        buf = g.serialize(bytearray(), prev)
        res = unpack_from(igmpv3_report._PACK_STR, bytes(buf))
        buf = bytearray(buf)
        pack_into('!H', buf, 2, 0)
        self.assertEqual(res[0], IGMP_TYPE_REPORT_V3)
        self.assertEqual(res[1], checksum(buf))
        self.assertEqual(res[2], len(records))

    def test_json(self):
        jsondict = self.g.to_jsondict()
        g = igmpv3_report.from_jsondict(jsondict['igmpv3_report'])
        self.assertEqual(str(self.g), str(g))

    def test_json_with_records(self):
        self.setUp_with_records()
        self.test_json()