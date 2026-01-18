import os
import socket
import sys
import unittest
from unittest import mock
from os_ken.lib import pcaplib
from os_ken.lib.packet import packet
from os_ken.lib.packet import zebra
from os_ken.utils import binary_str
def test_pcap_quagga(self):
    files = ['zebra_v2', 'zebra_v3']
    for f in files:
        self._test_pcap_single(f)