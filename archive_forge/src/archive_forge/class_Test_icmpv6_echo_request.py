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
class Test_icmpv6_echo_request(unittest.TestCase):
    type_ = 128
    code = 0
    csum = 42354
    id_ = 30240
    seq = 0
    data = b'\x01\xc9\xe76\xd39\x06\x00'
    buf = b'\x80\x00\xa5rv \x00\x00'

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        echo = icmpv6.echo(0, 0)
        self.assertEqual(echo.id, 0)
        self.assertEqual(echo.seq, 0)
        self.assertEqual(echo.data, None)

    def _test_parser(self, data=None):
        buf = self.buf + (data or b'')
        msg, n, _ = icmpv6.icmpv6.parser(buf)
        self.assertEqual(msg.type_, self.type_)
        self.assertEqual(msg.code, self.code)
        self.assertEqual(msg.csum, self.csum)
        self.assertEqual(msg.data.id, self.id_)
        self.assertEqual(msg.data.seq, self.seq)
        self.assertEqual(msg.data.data, data)
        self.assertEqual(n, None)

    def test_parser_without_data(self):
        self._test_parser()

    def test_parser_with_data(self):
        self._test_parser(self.data)

    def _test_serialize(self, echo_data=None):
        buf = self.buf + (echo_data or b'')
        src_ipv6 = '3ffe:507:0:1:200:86ff:fe05:80da'
        dst_ipv6 = '3ffe:501:0:1001::2'
        prev = ipv6(6, 0, 0, len(buf), 64, 255, src_ipv6, dst_ipv6)
        echo_csum = icmpv6_csum(prev, buf)
        echo = icmpv6.echo(self.id_, self.seq, echo_data)
        icmp = icmpv6.icmpv6(self.type_, self.code, 0, echo)
        buf = bytes(icmp.serialize(bytearray(), prev))
        type_, code, csum = struct.unpack_from(icmp._PACK_STR, buf, 0)
        id_, seq = struct.unpack_from(echo._PACK_STR, buf, icmp._MIN_LEN)
        data = buf[icmp._MIN_LEN + echo._MIN_LEN:]
        data = data if len(data) != 0 else None
        self.assertEqual(type_, self.type_)
        self.assertEqual(code, self.code)
        self.assertEqual(csum, echo_csum)
        self.assertEqual(id_, self.id_)
        self.assertEqual(seq, self.seq)
        self.assertEqual(data, echo_data)

    def test_serialize_without_data(self):
        self._test_serialize()

    def test_serialize_with_data(self):
        self._test_serialize(self.data)

    def test_to_string(self):
        ec = icmpv6.echo(self.id_, self.seq, self.data)
        ic = icmpv6.icmpv6(self.type_, self.code, self.csum, ec)
        echo_values = {'id': self.id_, 'seq': self.seq, 'data': self.data}
        _echo_str = ','.join(['%s=%s' % (k, repr(echo_values[k])) for k, v in inspect.getmembers(ec) if k in echo_values])
        echo_str = '%s(%s)' % (icmpv6.echo.__name__, _echo_str)
        icmp_values = {'type_': repr(self.type_), 'code': repr(self.code), 'csum': repr(self.csum), 'data': echo_str}
        _ic_str = ','.join(['%s=%s' % (k, icmp_values[k]) for k, v in inspect.getmembers(ic) if k in icmp_values])
        ic_str = '%s(%s)' % (icmpv6.icmpv6.__name__, _ic_str)
        self.assertEqual(str(ic), ic_str)
        self.assertEqual(repr(ic), ic_str)

    def test_default_args(self):
        prev = ipv6(nxt=inet.IPPROTO_ICMPV6)
        ic = icmpv6.icmpv6(type_=icmpv6.ICMPV6_ECHO_REQUEST, data=icmpv6.echo())
        prev.serialize(ic, None)
        buf = ic.serialize(bytearray(), prev)
        res = struct.unpack(icmpv6.icmpv6._PACK_STR, bytes(buf[:4]))
        self.assertEqual(res[0], icmpv6.ICMPV6_ECHO_REQUEST)
        self.assertEqual(res[1], 0)
        self.assertEqual(res[2], icmpv6_csum(prev, buf))
        res = struct.unpack(icmpv6.echo._PACK_STR, bytes(buf[4:]))
        self.assertEqual(res[0], 0)
        self.assertEqual(res[1], 0)

    def test_json(self):
        ec = icmpv6.echo(self.id_, self.seq, self.data)
        ic1 = icmpv6.icmpv6(self.type_, self.code, self.csum, ec)
        jsondict = ic1.to_jsondict()
        ic2 = icmpv6.icmpv6.from_jsondict(jsondict['icmpv6'])
        self.assertEqual(str(ic1), str(ic2))