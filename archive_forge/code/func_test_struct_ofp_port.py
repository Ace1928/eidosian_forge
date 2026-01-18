import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_port(self):
    self.assertEqual(OFP_PORT_PACK_STR, '!I4x6s2x16sIIIIIIII')
    self.assertEqual(OFP_PORT_SIZE, 64)