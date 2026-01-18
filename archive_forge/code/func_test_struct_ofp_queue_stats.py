import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_stats(self):
    self.assertEqual(OFP_QUEUE_STATS_PACK_STR, '!IIQQQ')
    self.assertEqual(OFP_QUEUE_STATS_SIZE, 32)