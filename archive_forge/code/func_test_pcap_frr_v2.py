import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
@_patch_frr_v2
def test_pcap_frr_v2(self):
    files = ['zebra_v4_frr_v2']
    for f in files:
        self._test_pcap_single(f)