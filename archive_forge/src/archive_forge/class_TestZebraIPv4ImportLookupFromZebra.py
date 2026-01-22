import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraIPv4ImportLookupFromZebra(unittest.TestCase):
    buf = b'\xc0\xa8\x01\x01\x00\x00\x00\x14\x01\x01\x00\x00\x00\x02'
    prefix = '192.168.1.1'
    metric = 20
    nexthop_num = 1
    nexthop_type = zebra.ZEBRA_NEXTHOP_IFINDEX
    ifindex = 2
    from_zebra = True

    def test_parser(self):
        body = zebra.ZebraIPv4ImportLookup.parse_from_zebra(self.buf)
        self.assertEqual(self.prefix, body.prefix)
        self.assertEqual(self.metric, body.metric)
        self.assertEqual(self.nexthop_num, len(body.nexthops))
        self.assertEqual(self.nexthop_type, body.nexthops[0].type)
        self.assertEqual(self.ifindex, body.nexthops[0].ifindex)
        self.assertEqual(self.from_zebra, body.from_zebra)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))