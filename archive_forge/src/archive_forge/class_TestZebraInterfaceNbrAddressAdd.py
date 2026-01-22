import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraInterfaceNbrAddressAdd(unittest.TestCase):
    buf = b'\x00\x00\x00\x01\x02\xc0\xa8\x01\x00\x18'
    ifindex = 1
    family = socket.AF_INET
    prefix = '192.168.1.0/24'

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraInterfaceNbrAddressAdd.parse(self.buf)
        self.assertEqual(self.ifindex, body.ifindex)
        self.assertEqual(self.family, body.family)
        self.assertEqual(self.prefix, body.prefix)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))