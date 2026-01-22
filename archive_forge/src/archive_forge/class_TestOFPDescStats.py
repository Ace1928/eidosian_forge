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
class TestOFPDescStats(unittest.TestCase):
    """ Test case for ofproto_v1_2_parser.OFPDescStats
    """
    mfr_desc = b'mfr_desc'.ljust(256)
    hw_desc = b'hw_desc'.ljust(256)
    sw_desc = b'sw_desc'.ljust(256)
    serial_num = b'serial_num'.ljust(32)
    dp_desc = b'dp_desc'.ljust(256)
    buf = mfr_desc + hw_desc + sw_desc + serial_num + dp_desc
    c = OFPDescStats(mfr_desc, hw_desc, sw_desc, serial_num, dp_desc)

    def test_init(self):
        self.assertEqual(self.mfr_desc, self.c.mfr_desc)
        self.assertEqual(self.hw_desc, self.c.hw_desc)
        self.assertEqual(self.sw_desc, self.c.sw_desc)
        self.assertEqual(self.serial_num, self.c.serial_num)
        self.assertEqual(self.dp_desc, self.c.dp_desc)

    def test_parser(self):
        res = self.c.parser(self.buf, 0)
        self.assertEqual(self.mfr_desc, res.mfr_desc)
        self.assertEqual(self.hw_desc, res.hw_desc)
        self.assertEqual(self.sw_desc, res.sw_desc)
        self.assertEqual(self.serial_num, res.serial_num)
        self.assertEqual(self.dp_desc, res.dp_desc)