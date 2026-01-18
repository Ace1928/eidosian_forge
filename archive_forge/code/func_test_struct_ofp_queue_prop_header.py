import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_queue_prop_header(self):
    self.assertEqual(OFP_QUEUE_PROP_HEADER_PACK_STR, '!HH4x')
    self.assertEqual(OFP_QUEUE_PROP_HEADER_SIZE, 8)