import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_serialize_with_compression(self):
    nxt = 0
    size = 3
    type_ = 3
    seg = 0
    cmpi = 8
    cmpe = 8
    adrs = ['2001:db8:dead::1', '2001:db8:dead::2', '2001:db8:dead::3']
    pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
    slice_i = slice(cmpi, 16)
    slice_e = slice(cmpe, 16)
    routing = ipv6.routing_type3(nxt, size, type_, seg, cmpi, cmpe, adrs)
    buf = routing.serialize()
    form = '!BBBBBB2x8s8s8s'
    res = struct.unpack_from(form, bytes(buf))
    self.assertEqual(nxt, res[0])
    self.assertEqual(size, res[1])
    self.assertEqual(type_, res[2])
    self.assertEqual(seg, res[3])
    self.assertEqual(cmpi, res[4] >> 4)
    self.assertEqual(cmpe, res[4] & 15)
    self.assertEqual(pad, res[5])
    self.assertEqual(addrconv.ipv6.text_to_bin(adrs[0])[slice_i], res[6])
    self.assertEqual(addrconv.ipv6.text_to_bin(adrs[1])[slice_i], res[7])
    self.assertEqual(addrconv.ipv6.text_to_bin(adrs[2])[slice_e], res[8])