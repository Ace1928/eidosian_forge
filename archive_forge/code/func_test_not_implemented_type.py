import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_not_implemented_type(self):
    not_implemented_buf = struct.pack('!BBBBBB2x', 0, 6, ipv6.routing.ROUTING_TYPE_2, 0, 0, 0)
    instance = ipv6.routing.parser(not_implemented_buf)
    assert None is instance