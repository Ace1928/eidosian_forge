import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_serialize_with_multi_headers(self):
    self.setUp_with_multi_headers()
    self.test_serialize()
    data = bytearray()
    prev = None
    buf = self.ip.serialize(data, prev)
    offset = ipv6.ipv6._MIN_LEN
    hop_opts = ipv6.hop_opts.parser(bytes(buf[offset:]))
    offset += len(hop_opts)
    auth = ipv6.auth.parser(bytes(buf[offset:]))
    self.assertEqual(repr(self.hop_opts), repr(hop_opts))
    self.assertEqual(repr(self.auth), repr(auth))