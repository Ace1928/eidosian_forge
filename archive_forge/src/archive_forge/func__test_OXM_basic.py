import unittest
import logging
from os_ken.ofproto.ofproto_v1_2 import *
def _test_OXM_basic(self, value, field, hasmask, length):
    self._test_OXM(value, OFPXMC_OPENFLOW_BASIC, field, hasmask, length)