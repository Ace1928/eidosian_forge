import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_serialize_with_dst_opts(self):
    self.setUp_with_dst_opts()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ip.serialize(data, prev)
    dst_opts = ipv6.dst_opts.parser(bytes(buf[ipv6.ipv6._MIN_LEN:]))
    self.assertEqual(repr(self.dst_opts), repr(dst_opts))