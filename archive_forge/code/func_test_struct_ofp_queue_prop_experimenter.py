import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_prop_experimenter(self):
    self.assertEqual(OFP_QUEUE_PROP_EXPERIMENTER_PACK_STR, '!I4x')
    self.assertEqual(OFP_QUEUE_PROP_EXPERIMENTER_SIZE, 16)