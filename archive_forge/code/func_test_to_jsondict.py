import logging
import unittest
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import vxlan
def test_to_jsondict(self):
    jsondict_from_pkt = self.pkt.to_jsondict()
    self.assertEqual(self.jsondict, jsondict_from_pkt)