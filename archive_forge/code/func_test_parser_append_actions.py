import unittest
import logging
from os_ken.ofproto.ofproto_v1_0_parser import *
from os_ken.ofproto.nx_actions import *
from os_ken.ofproto import ofproto_v1_0_parser
from os_ken.lib import addrconv
def test_parser_append_actions(self):
    res = self._parser(True).actions[0]
    self.assertEqual(self.action[self.ACTION_TYPE]['val'], res.type)
    self.assertEqual(self.action[self.ACTION_LEN]['val'], res.len)
    self.assertEqual(self.action[self.ACTION_PORT]['val'], res.port)
    self.assertEqual(self.action[self.ACTION_MAX_LEN]['val'], res.max_len)