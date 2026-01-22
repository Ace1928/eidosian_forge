import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraInterfaceBfdDestinationUpdate(unittest.TestCase):
    buf = b'\x00\x00\x00\x01\x02\xc0\xa8\x01\x01\x18\x04\x02\xc0\xa8\x01\x02\x18'
    ifindex = 1
    dst_family = socket.AF_INET
    dst_prefix = '192.168.1.1/24'
    status = zebra.BFD_STATUS_UP
    src_family = socket.AF_INET
    src_prefix = '192.168.1.2/24'

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraInterfaceBfdDestinationUpdate.parse(self.buf)
        self.assertEqual(self.ifindex, body.ifindex)
        self.assertEqual(self.dst_family, body.dst_family)
        self.assertEqual(self.dst_prefix, body.dst_prefix)
        self.assertEqual(self.status, body.status)
        self.assertEqual(self.src_family, body.src_family)
        self.assertEqual(self.src_prefix, body.src_prefix)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))