import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_set_field(self):
    self.assertEqual(OFP_ACTION_SET_FIELD_PACK_STR, '!HH4B')
    self.assertEqual(OFP_ACTION_SET_FIELD_SIZE, 8)