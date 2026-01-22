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
class TestOFPActionHeader(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionHeader
    """

    def test_init(self):
        type_ = ofproto.OFPAT_OUTPUT
        len_ = ofproto.OFP_ACTION_HEADER_SIZE
        fmt = ofproto.OFP_ACTION_HEADER_PACK_STR
        buf = pack(fmt, type_, len_)
        c = OFPActionHeader(type_, len_)
        self.assertEqual(type_, c.type)
        self.assertEqual(len_, c.len)

    def _test_serialize(self, type_, len_):
        fmt = ofproto.OFP_ACTION_HEADER_PACK_STR
        buf = pack(fmt, type_, len_)
        c = OFPActionHeader(type_, len_)
        buf = bytearray()
        c.serialize(buf, 0)
        fmt = ofproto.OFP_ACTION_HEADER_PACK_STR
        res = struct.unpack(fmt, bytes(buf))
        self.assertEqual(res[0], type_)
        self.assertEqual(res[1], len_)

    def test_serialize_mid(self):
        type_ = 11
        len_ = 8
        self._test_serialize(type_, len_)

    def test_serialize_max(self):
        type_ = 65535
        len_ = 65535
        self._test_serialize(type_, len_)

    def test_serialize_min(self):
        type_ = 0
        len_ = 0
        self._test_serialize(type_, len_)