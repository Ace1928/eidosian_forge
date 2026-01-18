import inspect
import logging
import struct
import unittest
from os_ken.lib import addrconv
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import sctp
from os_ken.ofproto import ether
from os_ken.ofproto import inet
def test_serialize_with_init_ack(self):
    self.setUp_with_init_ack()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_init_ack._PACK_STR, buf)
    self.assertEqual(sctp.chunk_init_ack.chunk_type(), res[0])
    self.assertEqual(self.flags, res[1])
    self.assertEqual(self.length, res[2])
    self.assertEqual(self.init_tag, res[3])
    self.assertEqual(self.a_rwnd, res[4])
    self.assertEqual(self.os, res[5])
    self.assertEqual(self.mis, res[6])
    self.assertEqual(self.i_tsn, res[7])
    buf = buf[sctp.chunk_init_ack._MIN_LEN:]
    res1 = struct.unpack_from(sctp.param_state_cookie._PACK_STR, buf)
    self.assertEqual(sctp.param_state_cookie.param_type(), res1[0])
    self.assertEqual(7, res1[1])
    self.assertEqual(b'\x01\x02\x03', buf[sctp.param_state_cookie._MIN_LEN:sctp.param_state_cookie._MIN_LEN + 3])
    buf = buf[8:]
    res2 = struct.unpack_from(sctp.param_ipv4._PACK_STR, buf)
    self.assertEqual(sctp.param_ipv4.param_type(), res2[0])
    self.assertEqual(8, res2[1])
    self.assertEqual('192.168.1.1', addrconv.ipv4.bin_to_text(buf[sctp.param_ipv4._MIN_LEN:sctp.param_ipv4._MIN_LEN + 4]))
    buf = buf[8:]
    res3 = struct.unpack_from(sctp.param_ipv6._PACK_STR, buf)
    self.assertEqual(sctp.param_ipv6.param_type(), res3[0])
    self.assertEqual(20, res3[1])
    self.assertEqual('fe80::647e:1aff:fec4:8284', addrconv.ipv6.bin_to_text(buf[sctp.param_ipv6._MIN_LEN:sctp.param_ipv6._MIN_LEN + 16]))
    buf = buf[20:]
    res4 = struct.unpack_from(sctp.param_unrecognized_param._PACK_STR, buf)
    self.assertEqual(sctp.param_unrecognized_param.param_type(), res4[0])
    self.assertEqual(8, res4[1])
    self.assertEqual(b'\xff\xff\x00\x04', buf[sctp.param_unrecognized_param._MIN_LEN:sctp.param_unrecognized_param._MIN_LEN + 4])
    buf = buf[8:]
    res5 = struct.unpack_from(sctp.param_ecn._PACK_STR, buf)
    self.assertEqual(sctp.param_ecn.param_type(), res5[0])
    self.assertEqual(4, res5[1])
    buf = buf[4:]
    res6 = struct.unpack_from(sctp.param_host_addr._PACK_STR, buf)
    self.assertEqual(sctp.param_host_addr.param_type(), res6[0])
    self.assertEqual(14, res6[1])
    self.assertEqual(b'test host\x00', buf[sctp.param_host_addr._MIN_LEN:sctp.param_host_addr._MIN_LEN + 10])