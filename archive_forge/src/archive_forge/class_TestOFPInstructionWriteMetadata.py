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
class TestOFPInstructionWriteMetadata(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPInstructionWriteMetadata
    """
    type_ = ofproto.OFPIT_WRITE_METADATA
    len_ = ofproto.OFP_INSTRUCTION_WRITE_METADATA_SIZE
    metadata = 1302123111085380114
    metadata_mask = 18374966859414961920
    fmt = ofproto.OFP_INSTRUCTION_WRITE_METADATA_PACK_STR

    def test_init(self):
        c = OFPInstructionWriteMetadata(self.metadata, self.metadata_mask)
        self.assertEqual(self.type_, c.type)
        self.assertEqual(self.len_, c.len)
        self.assertEqual(self.metadata, c.metadata)
        self.assertEqual(self.metadata_mask, c.metadata_mask)

    def _test_parser(self, metadata, metadata_mask):
        buf = pack(self.fmt, self.type_, self.len_, metadata, metadata_mask)
        res = OFPInstructionWriteMetadata.parser(buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.metadata, metadata)
        self.assertEqual(res.metadata_mask, metadata_mask)

    def test_parser_metadata_mid(self):
        self._test_parser(self.metadata, self.metadata_mask)

    def test_parser_metadata_max(self):
        metadata = 18446744073709551615
        self._test_parser(metadata, self.metadata_mask)

    def test_parser_metadata_min(self):
        metadata = 0
        self._test_parser(metadata, self.metadata_mask)

    def test_parser_metadata_mask_max(self):
        metadata_mask = 18446744073709551615
        self._test_parser(self.metadata, metadata_mask)

    def test_parser_metadata_mask_min(self):
        metadata_mask = 0
        self._test_parser(self.metadata, metadata_mask)

    def _test_serialize(self, metadata, metadata_mask):
        c = OFPInstructionWriteMetadata(metadata, metadata_mask)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], metadata)
        self.assertEqual(res[3], metadata_mask)

    def test_serialize_metadata_mid(self):
        self._test_serialize(self.metadata, self.metadata_mask)

    def test_serialize_metadata_max(self):
        metadata = 18446744073709551615
        self._test_serialize(metadata, self.metadata_mask)

    def test_serialize_metadata_min(self):
        metadata = 0
        self._test_serialize(metadata, self.metadata_mask)

    def test_serialize_metadata_mask_max(self):
        metadata_mask = 18446744073709551615
        self._test_serialize(self.metadata, metadata_mask)

    def test_serialize_metadata_mask_min(self):
        metadata_mask = 0
        self._test_serialize(self.metadata, metadata_mask)