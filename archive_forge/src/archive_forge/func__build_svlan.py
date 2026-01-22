import unittest
import logging
import struct
from struct import *
from os_ken.ofproto import ether, inet
from os_ken.lib.packet.ethernet import ethernet
from os_ken.lib.packet.packet import Packet
from os_ken.lib.packet.ipv4 import ipv4
from os_ken.lib.packet.vlan import vlan
from os_ken.lib.packet.vlan import svlan
def _build_svlan(self):
    src_mac = '00:07:0d:af:f4:54'
    dst_mac = '00:00:00:00:00:00'
    ethertype = ether.ETH_TYPE_8021AD
    e = ethernet(dst_mac, src_mac, ethertype)
    pcp = 0
    cfi = 0
    vid = 32
    tci = pcp << 15 | cfi << 12 | vid
    ethertype = ether.ETH_TYPE_IP
    v = vlan(pcp, cfi, vid, ethertype)
    version = 4
    header_length = 20
    tos = 0
    total_length = 24
    identification = 35421
    flags = 0
    offset = 1480
    ttl = 64
    proto = inet.IPPROTO_ICMP
    csum = 42994
    src = '131.151.32.21'
    dst = '131.151.32.129'
    option = b'TEST'
    ip = ipv4(version, header_length, tos, total_length, identification, flags, offset, ttl, proto, csum, src, dst, option)
    p = Packet()
    p.add_protocol(e)
    p.add_protocol(self.sv)
    p.add_protocol(v)
    p.add_protocol(ip)
    p.serialize()
    return p