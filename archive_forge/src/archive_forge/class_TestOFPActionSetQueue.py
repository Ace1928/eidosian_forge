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
class TestOFPActionSetQueue(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPActionSetQueue
    """
    type_ = ofproto.OFPAT_SET_QUEUE
    len_ = ofproto.OFP_ACTION_SET_QUEUE_SIZE
    queue_id = 6606
    fmt = ofproto.OFP_ACTION_SET_QUEUE_PACK_STR

    def test_init(self):
        c = OFPActionSetQueue(self.queue_id)
        self.assertEqual(self.queue_id, c.queue_id)

    def _test_parser(self, queue_id):
        buf = pack(self.fmt, self.type_, self.len_, queue_id)
        res = OFPActionSetQueue.parser(buf, 0)
        self.assertEqual(res.len, self.len_)
        self.assertEqual(res.type, self.type_)
        self.assertEqual(res.queue_id, queue_id)

    def test_parser_mid(self):
        self._test_parser(self.queue_id)

    def test_parser_max(self):
        self._test_parser(4294967295)

    def test_parser_min(self):
        self._test_parser(0)

    def _test_serialize(self, queue_id):
        c = OFPActionSetQueue(queue_id)
        buf = bytearray()
        c.serialize(buf, 0)
        res = struct.unpack(self.fmt, bytes(buf))
        self.assertEqual(res[0], self.type_)
        self.assertEqual(res[1], self.len_)
        self.assertEqual(res[2], queue_id)

    def test_serialize_mid(self):
        self._test_serialize(self.queue_id)

    def test_serialize_max(self):
        self._test_serialize(4294967295)

    def test_serialize_min(self):
        self._test_serialize(0)