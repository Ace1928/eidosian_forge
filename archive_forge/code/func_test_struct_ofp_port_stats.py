import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_port_stats(self):
    self.assertEqual(OFP_PORT_STATS_PACK_STR, '!I4xQQQQQQQQQQQQ')
    self.assertEqual(OFP_PORT_STATS_SIZE, 104)