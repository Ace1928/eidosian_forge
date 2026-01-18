import inspect
import logging
import struct
import unittest
from os_ken.lib import addrconv
from os_ken.lib.packet import dhcp
def test_parser_corrupted(self):
    corrupt_buf = self.buf[:-4]
    pkt, _, rest = dhcp.dhcp.parser(corrupt_buf)
    self.assertTrue(isinstance(pkt, dhcp.dhcp))
    self.assertTrue(isinstance(pkt.options, dhcp.options))
    for opt in pkt.options.option_list[:-1]:
        self.assertTrue(isinstance(opt, dhcp.option))
    self.assertTrue(isinstance(pkt.options.option_list[-1], bytes))
    buf = pkt.serialize()
    self.assertEqual(str(buf), str(corrupt_buf))
    self.assertEqual(b'', rest)