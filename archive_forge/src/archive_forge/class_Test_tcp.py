import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import inet
from os_ken.lib.packet import tcp
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet import packet_utils
from os_ken.lib import addrconv
class Test_tcp(unittest.TestCase):
    """ Test case for tcp
    """
    src_port = 6431
    dst_port = 8080
    seq = 5
    ack = 1
    offset = 6
    bits = 42
    window_size = 2048
    csum = 12345
    urgent = 128
    option = b'\x01\x02\x03\x04'
    t = tcp.tcp(src_port, dst_port, seq, ack, offset, bits, window_size, csum, urgent, option)
    buf = pack(tcp.tcp._PACK_STR, src_port, dst_port, seq, ack, offset << 4, bits, window_size, csum, urgent)
    buf += option

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.src_port, self.t.src_port)
        self.assertEqual(self.dst_port, self.t.dst_port)
        self.assertEqual(self.seq, self.t.seq)
        self.assertEqual(self.ack, self.t.ack)
        self.assertEqual(self.offset, self.t.offset)
        self.assertEqual(self.bits, self.t.bits)
        self.assertEqual(self.window_size, self.t.window_size)
        self.assertEqual(self.csum, self.t.csum)
        self.assertEqual(self.urgent, self.t.urgent)
        self.assertEqual(self.option, self.t.option)

    def test_parser(self):
        r1, r2, _ = self.t.parser(self.buf)
        self.assertEqual(self.src_port, r1.src_port)
        self.assertEqual(self.dst_port, r1.dst_port)
        self.assertEqual(self.seq, r1.seq)
        self.assertEqual(self.ack, r1.ack)
        self.assertEqual(self.offset, r1.offset)
        self.assertEqual(self.bits, r1.bits)
        self.assertEqual(self.window_size, r1.window_size)
        self.assertEqual(self.csum, r1.csum)
        self.assertEqual(self.urgent, r1.urgent)
        self.assertEqual(self.option, r1.option)
        self.assertEqual(None, r2)

    def test_serialize(self):
        offset = 5
        csum = 0
        src_ip = '192.168.10.1'
        dst_ip = '192.168.100.1'
        prev = ipv4(4, 5, 0, 0, 0, 0, 0, 64, inet.IPPROTO_TCP, 0, src_ip, dst_ip)
        t = tcp.tcp(self.src_port, self.dst_port, self.seq, self.ack, offset, self.bits, self.window_size, csum, self.urgent)
        buf = t.serialize(bytearray(), prev)
        res = struct.unpack(tcp.tcp._PACK_STR, bytes(buf))
        self.assertEqual(res[0], self.src_port)
        self.assertEqual(res[1], self.dst_port)
        self.assertEqual(res[2], self.seq)
        self.assertEqual(res[3], self.ack)
        self.assertEqual(res[4], offset << 4)
        self.assertEqual(res[5], self.bits)
        self.assertEqual(res[6], self.window_size)
        self.assertEqual(res[8], self.urgent)
        self.assertEqual(offset * 4, len(t))
        ph = struct.pack('!4s4sBBH', addrconv.ipv4.text_to_bin(src_ip), addrconv.ipv4.text_to_bin(dst_ip), 0, 6, offset * 4)
        d = ph + buf
        s = packet_utils.checksum(d)
        self.assertEqual(0, s)

    def test_serialize_option(self):
        offset = 0
        csum = 0
        option = [tcp.TCPOptionMaximumSegmentSize(max_seg_size=1460), tcp.TCPOptionSACKPermitted(), tcp.TCPOptionTimestamps(ts_val=287454020, ts_ecr=1432778632), tcp.TCPOptionNoOperation(), tcp.TCPOptionWindowScale(shift_cnt=9)]
        option_buf = b'\x02\x04\x05\xb4\x04\x02\x08\n\x11"3DUfw\x88\x01\x03\x03\t'
        prev = ipv4(4, 5, 0, 0, 0, 0, 0, 64, inet.IPPROTO_TCP, 0, '192.168.10.1', '192.168.100.1')
        t = tcp.tcp(self.src_port, self.dst_port, self.seq, self.ack, offset, self.bits, self.window_size, csum, self.urgent, option)
        buf = t.serialize(bytearray(), prev)
        r_option_buf = buf[tcp.tcp._MIN_LEN:tcp.tcp._MIN_LEN + len(option_buf)]
        self.assertEqual(option_buf, r_option_buf)
        r_tcp, _, _ = tcp.tcp.parser(buf)
        self.assertEqual(str(option), str(r_tcp.option))

    def test_malformed_tcp(self):
        m_short_buf = self.buf[1:tcp.tcp._MIN_LEN]
        self.assertRaises(Exception, tcp.tcp.parser, m_short_buf)

    def test_default_args(self):
        prev = ipv4(proto=inet.IPPROTO_TCP)
        t = tcp.tcp()
        buf = t.serialize(bytearray(), prev)
        res = struct.unpack(tcp.tcp._PACK_STR, buf)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 5 << 4)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        self.assertEqual(res[8], 0)
        t = tcp.tcp(option=[tcp.TCPOptionMaximumSegmentSize(1460)])
        buf = t.serialize(bytearray(), prev)
        res = struct.unpack(tcp.tcp._PACK_STR + '4s', buf)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 6 << 4)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        self.assertEqual(res[8], 0)
        self.assertEqual(res[9], b'\x02\x04\x05\xb4')
        t = tcp.tcp(offset=7, option=[tcp.TCPOptionWindowScale(shift_cnt=9)])
        buf = t.serialize(bytearray(), prev)
        res = struct.unpack(tcp.tcp._PACK_STR + '8s', buf)
        self.assertEqual(res[0], 1)
        self.assertEqual(res[1], 1)
        self.assertEqual(res[2], 0)
        self.assertEqual(res[3], 0)
        self.assertEqual(res[4], 7 << 4)
        self.assertEqual(res[5], 0)
        self.assertEqual(res[6], 0)
        self.assertEqual(res[8], 0)
        self.assertEqual(res[9], b'\x03\x03\t\x00\x00\x00\x00\x00')

    def test_json(self):
        jsondict = self.t.to_jsondict()
        t = tcp.tcp.from_jsondict(jsondict['tcp'])
        self.assertEqual(str(self.t), str(t))