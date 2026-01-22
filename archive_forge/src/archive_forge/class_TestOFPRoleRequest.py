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
class TestOFPRoleRequest(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPRoleRequest
    """
    role = 2147483648
    generation_id = 1270985291017894273

    def test_init(self):
        c = OFPRoleRequest(_Datapath, self.role, self.generation_id)
        self.assertEqual(self.role, c.role)
        self.assertEqual(self.generation_id, c.generation_id)

    def _test_serialize(self, role, generation_id):
        c = OFPRoleRequest(_Datapath, role, generation_id)
        c.serialize()
        self.assertEqual(ofproto.OFP_VERSION, c.version)
        self.assertEqual(ofproto.OFPT_ROLE_REQUEST, c.msg_type)
        self.assertEqual(0, c.xid)
        fmt = '!' + ofproto.OFP_HEADER_PACK_STR.replace('!', '') + ofproto.OFP_ROLE_REQUEST_PACK_STR.replace('!', '')
        res = struct.unpack(fmt, bytes(c.buf))
        self.assertEqual(ofproto.OFP_VERSION, res[0])
        self.assertEqual(ofproto.OFPT_ROLE_REQUEST, res[1])
        self.assertEqual(len(c.buf), res[2])
        self.assertEqual(0, res[3])
        self.assertEqual(role, res[4])
        self.assertEqual(generation_id, res[5])

    def test_serialize_mid(self):
        self._test_serialize(self.role, self.generation_id)

    def test_serialize_max(self):
        role = 4294967295
        generation_id = 18446744073709551615
        self._test_serialize(role, generation_id)

    def test_serialize_min(self):
        role = 0
        generation_id = 0
        self._test_serialize(role, generation_id)

    def test_serialize_p1(self):
        role = ofproto.OFPCR_ROLE_EQUAL
        self._test_serialize(role, self.generation_id)

    def test_serialize_p2(self):
        role = ofproto.OFPCR_ROLE_MASTER
        self._test_serialize(role, self.generation_id)

    def test_serialize_p3(self):
        role = ofproto.OFPCR_ROLE_SLAVE
        self._test_serialize(role, self.generation_id)