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
class TestOFPActionSetMplsTtl(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionSetMplsTtl
    """
    type_ = ofproto.OFPAT_SET_MPLS_TTL
    len_ = ofproto.OFP_ACTION_MPLS_TTL_SIZE
    mpls_ttl = 254
    fmt = ofproto.OFP_ACTION_MPLS_TTL_PACK_STR

    def test_init(self):
        c = OFPActionSetMplsTtl(self.mpls_ttl)
        self.assertEqual(self.mpls_ttl, c.mpls_ttl)

    def _test_parser(self, mpls_ttl):
        buf = pack(self.fmt, self.type_, self.len_, mpls_ttl)
        res = OFPActionSetMplsTtl.parser(buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.mpls_ttl, mpls_ttl)

    def test_parser_mid(self):
        self._test_parser(self.mpls_ttl)

    def test_parser_max(self):
        self._test_parser(255)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, mpls_ttl):
        c = OFPActionSetMplsTtl(mpls_ttl)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], mpls_ttl)

    def test_serialize_mid(self):
        self._test_serialize(self.mpls_ttl)

    def test_serialize_max(self):
        self._test_serialize(255)

    def test_serialize_min(self):
        self._test_serialize(0)