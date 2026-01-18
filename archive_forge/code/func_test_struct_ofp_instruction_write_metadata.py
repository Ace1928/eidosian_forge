import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_instruction_write_metadata(self):
    self.assertEqual(OFP_INSTRUCTION_WRITE_METADATA_PACK_STR, '!HH4xQQ')
    self.assertEqual(OFP_INSTRUCTION_WRITE_METADATA_SIZE, 24)