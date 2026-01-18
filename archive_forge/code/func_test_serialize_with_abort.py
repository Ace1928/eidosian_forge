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
def test_serialize_with_abort(self):
    self.setUp_with_abort()
    buf = self._test_serialize()
    res = struct.unpack_from(sctp.chunk_abort._PACK_STR, buf)
    self.assertEqual(sctp.chunk_abort.chunk_type(), res[0])
    flags = self.tflag << 0
    self.assertEqual(flags, res[1])
    self.assertEqual(self.length, res[2])
    buf = buf[sctp.chunk_abort._MIN_LEN:]
    res1 = struct.unpack_from(sctp.cause_invalid_stream_id._PACK_STR, buf)
    self.assertEqual(sctp.cause_invalid_stream_id.cause_code(), res1[0])
    self.assertEqual(8, res1[1])
    self.assertEqual(4096, res1[2])
    buf = buf[8:]
    res2 = struct.unpack_from(sctp.cause_missing_param._PACK_STR, buf)
    self.assertEqual(sctp.cause_missing_param.cause_code(), res2[0])
    self.assertEqual(16, res2[1])
    self.assertEqual(4, res2[2])
    types = []
    for count in range(4):
        tmp, = struct.unpack_from('!H', buf, sctp.cause_missing_param._MIN_LEN + 2 * count)
        types.append(tmp)
    self.assertEqual(str([sctp.PTYPE_IPV4, sctp.PTYPE_IPV6, sctp.PTYPE_COOKIE_PRESERVE, sctp.PTYPE_HOST_ADDR]), str(types))
    buf = buf[16:]
    res3 = struct.unpack_from(sctp.cause_stale_cookie._PACK_STR, buf)
    self.assertEqual(sctp.cause_stale_cookie.cause_code(), res3[0])
    self.assertEqual(8, res3[1])
    self.assertEqual(b'\x00\x00\x13\x88', buf[sctp.cause_stale_cookie._MIN_LEN:sctp.cause_stale_cookie._MIN_LEN + 4])
    buf = buf[8:]
    res4 = struct.unpack_from(sctp.cause_out_of_resource._PACK_STR, buf)
    self.assertEqual(sctp.cause_out_of_resource.cause_code(), res4[0])
    self.assertEqual(4, res4[1])
    buf = buf[4:]
    res5 = struct.unpack_from(sctp.cause_unresolvable_addr._PACK_STR, buf)
    self.assertEqual(sctp.cause_unresolvable_addr.cause_code(), res5[0])
    self.assertEqual(20, res5[1])
    self.assertEqual(b'\x00\x0b\x00\x0etest' + b' host\x00\x00\x00', buf[sctp.cause_unresolvable_addr._MIN_LEN:sctp.cause_unresolvable_addr._MIN_LEN + 16])
    buf = buf[20:]
    res6 = struct.unpack_from(sctp.cause_unrecognized_chunk._PACK_STR, buf)
    self.assertEqual(sctp.cause_unrecognized_chunk.cause_code(), res6[0])
    self.assertEqual(8, res6[1])
    self.assertEqual(b'\xff\x00\x00\x04', buf[sctp.cause_unrecognized_chunk._MIN_LEN:sctp.cause_unrecognized_chunk._MIN_LEN + 4])
    buf = buf[8:]
    res7 = struct.unpack_from(sctp.cause_invalid_param._PACK_STR, buf)
    self.assertEqual(sctp.cause_invalid_param.cause_code(), res7[0])
    self.assertEqual(4, res7[1])
    buf = buf[4:]
    res8 = struct.unpack_from(sctp.cause_unrecognized_param._PACK_STR, buf)
    self.assertEqual(sctp.cause_unrecognized_param.cause_code(), res8[0])
    self.assertEqual(8, res8[1])
    self.assertEqual(b'\xff\xff\x00\x04', buf[sctp.cause_unrecognized_param._MIN_LEN:sctp.cause_unrecognized_param._MIN_LEN + 4])
    buf = buf[8:]
    res9 = struct.unpack_from(sctp.cause_no_userdata._PACK_STR, buf)
    self.assertEqual(sctp.cause_no_userdata.cause_code(), res9[0])
    self.assertEqual(8, res9[1])
    self.assertEqual(b'\x00\x01\xe2@', buf[sctp.cause_no_userdata._MIN_LEN:sctp.cause_no_userdata._MIN_LEN + 4])
    buf = buf[8:]
    res10 = struct.unpack_from(sctp.cause_cookie_while_shutdown._PACK_STR, buf)
    self.assertEqual(sctp.cause_cookie_while_shutdown.cause_code(), res10[0])
    self.assertEqual(4, res10[1])
    buf = buf[4:]
    res11 = struct.unpack_from(sctp.cause_restart_with_new_addr._PACK_STR, buf)
    self.assertEqual(sctp.cause_restart_with_new_addr.cause_code(), res11[0])
    self.assertEqual(12, res11[1])
    self.assertEqual(b'\x00\x05\x00\x08\xc0\xa8\x01\x01', buf[sctp.cause_restart_with_new_addr._MIN_LEN:sctp.cause_restart_with_new_addr._MIN_LEN + 8])
    buf = buf[12:]
    res12 = struct.unpack_from(sctp.cause_user_initiated_abort._PACK_STR, buf)
    self.assertEqual(sctp.cause_user_initiated_abort.cause_code(), res12[0])
    self.assertEqual(19, res12[1])
    self.assertEqual(b'Key Interrupt.\x00', buf[sctp.cause_user_initiated_abort._MIN_LEN:sctp.cause_user_initiated_abort._MIN_LEN + 15])
    buf = buf[20:]
    res13 = struct.unpack_from(sctp.cause_protocol_violation._PACK_STR, buf)
    self.assertEqual(sctp.cause_protocol_violation.cause_code(), res13[0])
    self.assertEqual(20, res13[1])
    self.assertEqual(b'Unknown reason.\x00', buf[sctp.cause_protocol_violation._MIN_LEN:sctp.cause_protocol_violation._MIN_LEN + 16])