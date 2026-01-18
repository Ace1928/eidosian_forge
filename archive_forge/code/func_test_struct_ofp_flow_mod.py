import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_flow_mod(self):
    self.assertEqual(OFP_FLOW_MOD_PACK_STR, '!QQBBHHHIIIH2xHHBBBB')
    self.assertEqual(OFP_FLOW_MOD_SIZE, 56)