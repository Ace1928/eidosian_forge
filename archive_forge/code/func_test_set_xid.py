import binascii
import unittest
import struct
from os_ken import exception
from os_ken.ofproto import ofproto_common, ofproto_parser
from os_ken.ofproto import ofproto_v1_0, ofproto_v1_0_parser
import logging
def test_set_xid(self):
    xid = 3841413783
    c = ofproto_parser.MsgBase(object)
    c.set_xid(xid)
    self.assertEqual(xid, c.xid)