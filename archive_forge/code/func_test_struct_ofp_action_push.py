import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_push(self):
    self.assertEqual(OFP_ACTION_PUSH_PACK_STR, '!HHH2x')
    self.assertEqual(OFP_ACTION_PUSH_SIZE, 8)