import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
class TestZebraVrfAdd(unittest.TestCase):
    buf = b'VRF1\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
    vrf_name = 'VRF1'

    @_patch_frr_v2
    def test_parser(self):
        body = zebra.ZebraVrfAdd.parse(self.buf)
        self.assertEqual(self.vrf_name, body.vrf_name)
        buf = body.serialize()
        self.assertEqual(binary_str(self.buf), binary_str(buf))