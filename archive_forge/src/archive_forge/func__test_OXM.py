import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def _test_OXM(self, value, class_, field, hasmask, length):
    virfy = class_ << 16 | field << 9 | hasmask << 8 | length
    self.assertEqual(value >> 32, 0)
    self.assertEqual(value, virfy)