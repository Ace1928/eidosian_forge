import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_error_experimenter_msg(self):
    self.assertEqual(OFP_ERROR_EXPERIMENTER_MSG_PACK_STR, '!HHI')
    self.assertEqual(OFP_ERROR_EXPERIMENTER_MSG_SIZE, 16)