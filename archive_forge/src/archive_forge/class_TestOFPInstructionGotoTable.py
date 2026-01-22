import unittest
import logging
import socket
from struct import *
from os_ken.ofproto.ofproto_v1_2_parser import *
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ether
from os_ken.ofproto.ofproto_parser import MsgBase
from os_ken import utils
from os_ken.lib import addrconv
from os_ken.lib import pack_utils
class TestOFPInstructionGotoTable(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPInstructionGotoTable
    """
    type_ = ofproto.OFPIT_GOTO_TABLE
    len_ = ofproto.OFP_INSTRUCTION_GOTO_TABLE_SIZE
    fmt = ofproto.OFP_INSTRUCTION_GOTO_TABLE_PACK_STR

    def test_init(self):
        table_id = 3
        c = OFPInstructionGotoTable(table_id)
        self.assertEqual(self.type_, c.type)
        self.assertEqual(self.len_, c.len)
        self.assertEqual(table_id, c.table_id)

    def _test_parser(self, table_id):
        buf = pack(self.fmt, self.type_, self.len_, table_id)
        res = OFPInstructionGotoTable.parser(buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.table_id, table_id)

    def test_parser_mid(self):
        self._test_parser(3)

    def test_parser_max(self):
        self._test_parser(255)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, table_id):
        c = OFPInstructionGotoTable(table_id)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], table_id)

    def test_serialize_mid(self):
        self._test_serialize(3)

    def test_serialize_max(self):
        self._test_serialize(255)

    def test_serialize_min(self):
        self._test_serialize(0)