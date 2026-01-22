import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraRedistributeAdd(unittest.TestCase):
    buf = b'\x02'
    route_type = zebra.ZEBRA_ROUTE_CONNECT

    def test_parser(self):
        body = zebra.ZebraRedistributeAdd.parse(self.buf, version=3)
        self.assertEqual(self.route_type, body.route_type)
        buf = body.serialize(version=3)
        self.assertEqual(binary_str(self.buf), binary_str(buf))