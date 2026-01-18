import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_nw_ttl(self):
    self.assertEqual(OFP_ACTION_NW_TTL_PACK_STR, '!HHB3x')
    self.assertEqual(OFP_ACTION_NW_TTL_SIZE, 8)