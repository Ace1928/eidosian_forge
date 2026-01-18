import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_stats_reply(self):
    self.assertEqual(OFP_STATS_REPLY_PACK_STR, '!HH4x')
    self.assertEqual(OFP_STATS_REPLY_SIZE, 16)