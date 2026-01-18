import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_match(self):
    self.assertEqual(OFP_MATCH_PACK_STR, '!HHBBBB')
    self.assertEqual(OFP_MATCH_SIZE, 8)