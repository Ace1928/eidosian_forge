import unittest
import logging
import inspect
import struct
from os_ken.lib import addrconv
from os_ken.lib import ip
from os_ken.lib.packet import ipv6
def test_parser_with_adrs_zero(self):
    nxt = 0
    size = 0
    type_ = 3
    seg = 0
    cmpi = 0
    cmpe = 0
    adrs = []
    pad = (8 - ((len(adrs) - 1) * (16 - cmpi) + (16 - cmpe) % 8)) % 8
    form = '!BBBBBB2x'
    buf = struct.pack(form, nxt, size, type_, seg, cmpi << 4 | cmpe, pad << 4)
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