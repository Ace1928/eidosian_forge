import unittest
import logging
import struct
import inspect
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib import addrconv
def test_parse_with_auth_md5(self):
    buf = self.data_auth_md5
    pkt = packet.Packet(buf)
    i = iter(pkt)
    self.assertEqual(type(next(i)), ethernet.ethernet)
    self.assertEqual(type(next(i)), ipv4.ipv4)
    self.assertEqual(type(next(i)), udp.udp)
    bfd_obj = bfd.bfd.parser(next(i))[0]
    self.assertEqual(type(bfd_obj), bfd.bfd)
    self.assertEqual(type(bfd_obj.auth_cls), bfd.KeyedMD5)
    self.assertTrue(bfd_obj.authenticate(self.auth_keys))