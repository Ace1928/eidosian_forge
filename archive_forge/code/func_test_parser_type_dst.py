import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_type_dst(self):
    res = self.c.parser(self.buf, 0)
    self.assertEqual(self.dl_addr, res.dl_addr)