import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_aggregate_stats_request(self):
    self.assertEqual(OFP_AGGREGATE_STATS_REQUEST_PACK_STR, '!B3xII4xQQ')
    self.assertEqual(OFP_AGGREGATE_STATS_REQUEST_SIZE, 40)