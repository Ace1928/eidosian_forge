import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def test_parser_with_TimeExceeded(self):
    self.setUp_with_TimeExceeded()
    self.test_parser()