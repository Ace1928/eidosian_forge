import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_to_string_with_fragment(self):
    self.setUp_with_fragment()
    self.test_to_string()