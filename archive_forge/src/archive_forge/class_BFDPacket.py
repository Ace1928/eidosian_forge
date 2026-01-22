import logging
import time
import random
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.ofproto.ether import ETH_TYPE_IP, ETH_TYPE_ARP
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import inet
from os_ken.lib import hub
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import udp
from os_ken.lib.packet import bfd
from os_ken.lib.packet import arp
from os_ken.lib.packet.arp import ARP_REQUEST, ARP_REPLY
class BFDPacket(object):
    """
    BFDPacket class for parsing raw BFD packet, and generating BFD packet with
    Ethernet, IPv4, and UDP headers.
    """

    class BFDUnknownFormat(OSKenException):
        message = '%(msg)s'

    @staticmethod
    def bfd_packet(src_mac, dst_mac, src_ip, dst_ip, ipv4_id, src_port, dst_port, diag=0, state=0, flags=0, detect_mult=0, my_discr=0, your_discr=0, desired_min_tx_interval=0, required_min_rx_interval=0, required_min_echo_rx_interval=0, auth_cls=None):
        """
        Generate BFD packet with Ethernet/IPv4/UDP encapsulated.
        """
        pkt = packet.Packet()
        eth_pkt = ethernet.ethernet(dst_mac, src_mac, ETH_TYPE_IP)
        pkt.add_protocol(eth_pkt)
        ipv4_pkt = ipv4.ipv4(proto=inet.IPPROTO_UDP, src=src_ip, dst=dst_ip, tos=192, identification=ipv4_id, ttl=255)
        pkt.add_protocol(ipv4_pkt)
        udp_pkt = udp.udp(src_port=src_port, dst_port=dst_port)
        pkt.add_protocol(udp_pkt)
        bfd_pkt = bfd.bfd(ver=1, diag=diag, state=state, flags=flags, detect_mult=detect_mult, my_discr=my_discr, your_discr=your_discr, desired_min_tx_interval=desired_min_tx_interval, required_min_rx_interval=required_min_rx_interval, required_min_echo_rx_interval=required_min_echo_rx_interval, auth_cls=auth_cls)
        pkt.add_protocol(bfd_pkt)
        pkt.serialize()
        return pkt.data

    @staticmethod
    def bfd_parse(data):
        """
        Parse raw packet and return BFD class from packet library.
        """
        pkt = packet.Packet(data)
        i = iter(pkt)
        eth_pkt = next(i)
        assert isinstance(eth_pkt, ethernet.ethernet)
        ipv4_pkt = next(i)
        assert isinstance(ipv4_pkt, ipv4.ipv4)
        udp_pkt = next(i)
        assert isinstance(udp_pkt, udp.udp)
        udp_payload = next(i)
        return bfd.bfd.parser(udp_payload)[0]