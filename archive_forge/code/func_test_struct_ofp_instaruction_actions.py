import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_instaruction_actions(self):
    self.assertEqual(OFP_INSTRUCTION_ACTIONS_PACK_STR, '!HH4x')
    self.assertEqual(OFP_INSTRUCTION_ACTIONS_SIZE, 8)