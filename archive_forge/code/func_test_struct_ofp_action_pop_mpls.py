import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_action_pop_mpls(self):
    self.assertEqual(OFP_ACTION_POP_MPLS_PACK_STR, '!HHH2x')
    self.assertEqual(OFP_ACTION_POP_MPLS_SIZE, 8)