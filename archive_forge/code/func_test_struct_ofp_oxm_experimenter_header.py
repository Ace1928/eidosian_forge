import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_oxm_experimenter_header(self):
    self.assertEqual(OFP_OXM_EXPERIMENTER_HEADER_PACK_STR, '!II')
    self.assertEqual(OFP_OXM_EXPERIMENTER_HEADER_SIZE, 8)