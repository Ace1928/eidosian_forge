import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_output(self):
    self.assertEqual(OFP_ACTION_OUTPUT_PACK_STR, '!HHIH6x')
    self.assertEqual(OFP_ACTION_OUTPUT_SIZE, 16)