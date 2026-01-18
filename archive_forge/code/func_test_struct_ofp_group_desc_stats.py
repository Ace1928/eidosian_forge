import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_group_desc_stats(self):
    self.assertEqual(OFP_GROUP_DESC_STATS_PACK_STR, '!HBxI')
    self.assertEqual(OFP_GROUP_DESC_STATS_SIZE, 8)