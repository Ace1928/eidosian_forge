import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_header(self):
    self.assertEqual(OFP_HEADER_PACK_STR, '!BBHI')
    self.assertEqual(OFP_HEADER_SIZE, 8)