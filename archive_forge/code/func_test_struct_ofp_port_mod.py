import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def test_struct_ofp_port_mod(self):
    self.assertEqual(OFP_PORT_MOD_PACK_STR, '!I4x6s2xIII4x')
    self.assertEqual(OFP_PORT_MOD_SIZE, 40)