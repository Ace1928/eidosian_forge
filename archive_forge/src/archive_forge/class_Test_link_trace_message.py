import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib.packet import cfm
class Test_link_trace_message(unittest.TestCase):

    def setUp(self):
        self.md_lv = 1
        self.version = 1
        self.opcode = cfm.CFM_LINK_TRACE_MESSAGE
        self.use_fdb_only = 1
        self.first_tlv_offset = cfm.link_trace_message._TLV_OFFSET
        self.transaction_id = 12345
        self.ttl = 55
        self.ltm_orig_addr = '00:11:22:44:55:66'
        self.ltm_targ_addr = 'ab:cd:ef:23:12:65'
        self.tlvs = []
        self.end_tlv = 0
        self.ins = cfm.link_trace_message(self.md_lv, self.version, self.use_fdb_only, self.transaction_id, self.ttl, self.ltm_orig_addr, self.ltm_targ_addr, self.tlvs)
        self.form = '!4BIB6s6sB'
        self.buf = struct.pack(self.form, self.md_lv << 5 | self.version, self.opcode, self.use_fdb_only << 7, self.first_tlv_offset, self.transaction_id, self.ttl, addrconv.mac.text_to_bin(self.ltm_orig_addr), addrconv.mac.text_to_bin(self.ltm_targ_addr), self.end_tlv)

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.md_lv, self.ins.md_lv)
        self.assertEqual(self.version, self.ins.version)
        self.assertEqual(self.use_fdb_only, self.ins.use_fdb_only)
        self.assertEqual(self.transaction_id, self.ins.transaction_id)
        self.assertEqual(self.ttl, self.ins.ttl)
        self.assertEqual(self.ltm_orig_addr, self.ins.ltm_orig_addr)
        self.assertEqual(self.ltm_targ_addr, self.ins.ltm_targ_addr)
        self.assertEqual(self.tlvs, self.ins.tlvs)

    def test_parser(self):
        _res = cfm.link_trace_message.parser(self.buf)
        if type(_res) is tuple:
            res = _res[0]
        else:
            res = _res
        self.assertEqual(self.md_lv, res.md_lv)
        self.assertEqual(self.version, res.version)
        self.assertEqual(self.use_fdb_only, res.use_fdb_only)
        self.assertEqual(self.transaction_id, res.transaction_id)
        self.assertEqual(self.ttl, res.ttl)
        self.assertEqual(self.ltm_orig_addr, res.ltm_orig_addr)
        self.assertEqual(self.ltm_targ_addr, res.ltm_targ_addr)
        self.assertEqual(self.tlvs, res.tlvs)

    def test_serialize(self):
        buf = self.ins.serialize()
        res = struct.unpack_from(self.form, bytes(buf))
        self.assertEqual(self.md_lv, res[0] >> 5)
        self.assertEqual(self.version, res[0] & 31)
        self.assertEqual(self.opcode, res[1])
        self.assertEqual(self.use_fdb_only, res[2] >> 7)
        self.assertEqual(self.first_tlv_offset, res[3])
        self.assertEqual(self.transaction_id, res[4])
        self.assertEqual(self.ttl, res[5])
        self.assertEqual(addrconv.mac.text_to_bin(self.ltm_orig_addr), res[6])
        self.assertEqual(addrconv.mac.text_to_bin(self.ltm_targ_addr), res[7])
        self.assertEqual(self.end_tlv, res[8])

    def test_len(self):
        self.assertEqual(22, len(self.ins))

    def test_default_args(self):
        ins = cfm.link_trace_message()
        buf = ins.serialize()
        res = struct.unpack_from(cfm.link_trace_message._PACK_STR, bytes(buf))
        self.assertEqual(res[0] >> 5, 0)
        self.assertEqual(res[0] & 31, 0)
        self.assertEqual(res[1], 5)
        self.assertEqual(res[2] >> 7, 1)
        self.assertEqual(res[3], 17)
        self.assertEqual(res[4], 0)
        self.assertEqual(res[5], 64)
        self.assertEqual(res[6], addrconv.mac.text_to_bin('00:00:00:00:00:00'))
        self.assertEqual(res[7], addrconv.mac.text_to_bin('00:00:00:00:00:00'))