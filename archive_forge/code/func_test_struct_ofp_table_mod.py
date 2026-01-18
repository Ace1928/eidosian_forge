import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_table_mod(self):
    self.assertEqual(OFP_TABLE_MOD_PACK_STR, '!B3xI')
    self.assertEqual(OFP_TABLE_MOD_SIZE, 16)