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
def test_serialize_with_multi_chunks(self):
    self.setUp_with_multi_chunks()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_sack._PACK_STR, buf)
    self.assertEqual(sctp.chunk_sack.chunk_type(), res[0])
    self.assertEqual(self.s_flags, res[1])
    self.assertEqual(self.s_length, res[2])
    self.assertEqual(self.s_tsn_ack, res[3])
    self.assertEqual(self.s_a_rwnd, res[4])
    self.assertEqual(self.s_gapack_num, res[5])
    self.assertEqual(self.s_duptsn_num, res[6])
    buf = buf[self.s_length:]
    res = struct.unpack_from(sctp.chunk_data._PACK_STR, buf)
    self.assertEqual(sctp.chunk_data.chunk_type(), res[0])
    d1_flags = self.d1_unordered << 2 | self.d1_begin << 1 | self.d1_end << 0
    self.assertEqual(d1_flags, res[1])
    self.assertEqual(self.d1_length, res[2])
    self.assertEqual(self.d1_tsn, res[3])
    self.assertEqual(self.d1_sid, res[4])
    self.assertEqual(self.d1_seq, res[5])
    self.assertEqual(self.d1_payload_id, res[6])
    self.assertEqual(self.d1_payload_data, buf[sctp.chunk_data._MIN_LEN:sctp.chunk_data._MIN_LEN + 10])
    buf = buf[self.d1_length:]
    res = struct.unpack_from(sctp.chunk_data._PACK_STR, buf)
    self.assertEqual(sctp.chunk_data.chunk_type(), res[0])
    d2_flags = self.d2_unordered << 2 | self.d2_begin << 1 | self.d2_end << 0
    self.assertEqual(d2_flags, res[1])
    self.assertEqual(self.d2_length, res[2])
    self.assertEqual(self.d2_tsn, res[3])
    self.assertEqual(self.d2_sid, res[4])
    self.assertEqual(self.d2_seq, res[5])
    self.assertEqual(self.d2_payload_id, res[6])
    self.assertEqual(self.d2_payload_data, buf[sctp.chunk_data._MIN_LEN:sctp.chunk_data._MIN_LEN + 10])