import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_parser_with_compression(self):
    pass
    nxt = 0
    size = 3
    type_ = 3
    seg = 0
    cmpi = 8
    cmpe = 12
    adrs = ['2001:0db8:dead:0123:4567:89ab:cdef:0001', '2001:0db8:dead:0123:4567:89ab:cdef:0002', '2001:0db8:dead:0123:4567:89ab:cdef:0003']
    pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
    form = '!BBBBBB2x%ds%ds%ds' % (16 - cmpi, 16 - cmpi, 16 - cmpe)
    slice_i = slice(cmpi, 16)
    slice_e = slice(cmpe, 16)
    buf = struct.pack(form, nxt, size, type_, seg, cmpi << 4 | cmpe, pad << 4, addrconv.ipv6.text_to_bin(adrs[0])[slice_i], addrconv.ipv6.text_to_bin(adrs[1])[slice_i], addrconv.ipv6.text_to_bin(adrs[2])[slice_e])
    _res = ipv6.routing.parser(buf)
    if type(_res) is tuple:
        res = _res[0]
    else:
        res = _res
    self.assertEqual(nxt, res.nxt)
    self.assertEqual(size, res.size)
    self.assertEqual(type_, res.type_)
    self.assertEqual(seg, res.seg)
    self.assertEqual(cmpi, res.cmpi)
    self.assertEqual(cmpe, res.cmpe)
    self.assertEqual(pad, res._pad)
    self.assertEqual('::4567:89ab:cdef:1', res.adrs[0])
    self.assertEqual('::4567:89ab:cdef:2', res.adrs[1])
    self.assertEqual('::205.239.0.3', res.adrs[2])