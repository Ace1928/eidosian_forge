import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_instruction_goto_table(self):
    self.assertEqual(OFP_INSTRUCTION_GOTO_TABLE_PACK_STR, '!HHB3x')
    self.assertEqual(OFP_INSTRUCTION_GOTO_TABLE_SIZE, 8)