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
def test_serialize_with_auth_simple(self):
    pkt = packet.Packet()
    eth_pkt = ethernet.ethernet('08:00:27:d1:95:7c', '08:00:27:ed:54:41')
    pkt.add_protocol(eth_pkt)
    ip_pkt = ipv4.ipv4(src='192.168.57.2', dst='192.168.57.1', tos=192, identification=3216, proto=inet.IPPROTO_UDP)
    pkt.add_protocol(ip_pkt)
    udp_pkt = udp.udp(49152, 3784)
    pkt.add_protocol(udp_pkt)
    auth_cls = bfd.SimplePassword(auth_key_id=2, password=self.auth_keys[2])
    bfd_pkt = bfd.bfd(ver=1, diag=bfd.BFD_DIAG_NO_DIAG, flags=bfd.BFD_FLAG_AUTH_PRESENT, state=bfd.BFD_STATE_DOWN, detect_mult=3, my_discr=1, your_discr=0, desired_min_tx_interval=1000000, required_min_rx_interval=1000000, required_min_echo_rx_interval=0, auth_cls=auth_cls)
    pkt.add_protocol(bfd_pkt)
    self.assertEqual(len(pkt.protocols), 4)
    pkt.serialize()
    self.assertEqual(pkt.data, self.data_auth_simple)