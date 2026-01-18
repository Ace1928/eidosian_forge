import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_group_stats(self):
    self.assertEqual(OFP_GROUP_STATS_PACK_STR, '!H2xII4xQQ')
    self.assertEqual(OFP_GROUP_STATS_SIZE, 32)