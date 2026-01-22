import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraIPv4NexthopLookupMRib(unittest.TestCase):
    buf = b'\xc0\xa8\x01\x01'
    addr = '192.168.1.1'
    distance = None
    metric = None
    nexthop_num = 0

    def test_parser(self):
        body = zebra.ZebraIPv4NexthopLookupMRib.parse(self.buf)
        self.assertEqual(self.addr, body.addr)
        self.assertEqual(self.distance, body.distance)
        self.assertEqual(self.metric, body.metric)
        self.assertEqual(self.nexthop_num, len(body.nexthops))
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))