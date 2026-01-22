import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_link_trace_reply(unittest.TestCase):

    def setUp(self):
        self.md_lv = 1
        self.version = 1
        self.opcode = cfm.CFM_LINK_TRACE_REPLY
        self.use_fdb_only = 1
        self.fwd_yes = 0
        self.terminal_mep = 1
        self.first_tlv_offset = cfm.link_trace_reply._TLV_OFFSET
        self.transaction_id = 12345
        self.ttl = 55
        self.relay_action = 2
        self.ltm_orig_addr = '00:11:22:aa:bb:cc'
        self.ltm_targ_addr = '53:45:24:64:ac:ff'
        self.tlvs = []
        self.end_tlv = 0
        self.ins = cfm.link_trace_reply(self.md_lv, self.version, self.use_fdb_only, self.fwd_yes, self.terminal_mep, self.transaction_id, self.ttl, self.relay_action, self.tlvs)
        self.form = '!4BIBBB'
        self.buf = struct.pack(self.form, self.md_lv << 5 | self.version, self.opcode, self.use_fdb_only << 7 | self.fwd_yes << 6 | self.terminal_mep << 5, self.first_tlv_offset, self.transaction_id, self.ttl, self.relay_action, self.end_tlv)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.md_lv, self.ins.md_lv)
        self.assertEqual(self.version, self.ins.version)
        self.assertEqual(self.use_fdb_only, self.ins.use_fdb_only)
        self.assertEqual(self.fwd_yes, self.ins.fwd_yes)
        self.assertEqual(self.terminal_mep, self.ins.terminal_mep)
        self.assertEqual(self.transaction_id, self.ins.transaction_id)
        self.assertEqual(self.ttl, self.ins.ttl)
        self.assertEqual(self.relay_action, self.ins.relay_action)
        self.assertEqual(self.tlvs, self.ins.tlvs)

    def test_parser(self):
        _res = cfm.link_trace_reply.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.md_lv, res.md_lv)
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.use_fdb_only, self.ins.use_fdb_only)
        self.assertEqual(self.fwd_yes, self.ins.fwd_yes)
        self.assertEqual(self.terminal_mep, self.ins.terminal_mep)
        self.assertEqual(self.transaction_id, res.transaction_id)
        self.assertEqual(self.ttl, res.ttl)
        self.assertEqual(self.relay_action, res.relay_action)
        self.assertEqual(self.tlvs, res.tlvs)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.use_fdb_only, res[2] >> 7 & 1)
        self.assertEqual(self.fwd_yes, res[2] >> 6 & 1)
        self.assertEqual(self.terminal_mep, res[2] >> 5 & 1)
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.transaction_id, res[4])
        self.assertEqual(self.ttl, res[5])
        self.assertEqual(self.relay_action, res[6])
        self.assertEqual(self.end_tlv, res[7])

    def test_len(self):
        self.assertEqual(11, len(self.ins))

    def test_default_args(self):
        ins = cfm.link_trace_reply()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.link_trace_reply._PACK_STR, bytes(buf))
        self.assertEqual(res[0] >> 5, 0)
        self.assertEqual(res[0] & 31, 0)
        self.assertEqual(res[1], 4)
        self.assertEqual(res[2] >> 7, 1)
        self.assertEqual(res[2] >> 6 & 1, 0)
        self.assertEqual(res[2] >> 5 & 1, 1)
        self.assertEqual(res[3], 6)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 64)
        self.assertEqual(res[6], 1)