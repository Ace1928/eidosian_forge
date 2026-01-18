import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_sturct_ofp_stats_request(self):
    self.assertEqual(OFP_STATS_REQUEST_PACK_STR, '!HH4x')
    self.assertEqual(OFP_STATS_REQUEST_SIZE, 16)