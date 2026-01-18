import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_group_mod(self):
    self.assertEqual(OFP_GROUP_MOD_PACK_STR, '!HBxI')
    self.assertEqual(OFP_GROUP_MOD_SIZE, 16)