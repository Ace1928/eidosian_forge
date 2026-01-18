import inspect
import logging
import struct
import unittest
from os_ken.lib.packet import icmp
from os_ken.lib.packet import packet_utils
def test_serialize_with_dest_unreach(self):
    self.setUp_with_dest_unreach()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ic.serialize(data, prev)
    unreach = icmp.dest_unreach.parser(bytes(buf), icmp.icmp._MIN_LEN)
    self.assertEqual(repr(self.data), repr(unreach))