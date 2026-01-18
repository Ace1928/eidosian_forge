import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_aggregate_stats_reply(self):
    self.assertEqual(OFP_AGGREGATE_STATS_REPLY_PACK_STR, '!QQI4x')
    self.assertEqual(OFP_AGGREGATE_STATS_REPLY_SIZE, 24)