import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_sturct_ofp_table_stats(self):
    self.assertEqual(OFP_TABLE_STATS_PACK_STR, '!B7x32sQQIIQQQQIIIIQQ')
    self.assertEqual(OFP_TABLE_STATS_SIZE, 128)