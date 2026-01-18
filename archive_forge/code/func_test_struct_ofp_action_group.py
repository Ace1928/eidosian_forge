import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_group(self):
    self.assertEqual(OFP_ACTION_GROUP_PACK_STR, '!HHI')
    self.assertEqual(OFP_ACTION_GROUP_SIZE, 8)