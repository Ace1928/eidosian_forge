import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def test_to_string_with_echo(self):
    self.setUp_with_echo()
    self.test_to_string()