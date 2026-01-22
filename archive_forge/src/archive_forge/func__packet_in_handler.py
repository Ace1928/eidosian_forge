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
@set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
def _packet_in_handler(self, ev):
    msg = ev.msg
    datapath = msg.datapath
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    in_port = msg.match['in_port']
    pkt = packet.Packet(msg.data)
    if arp.arp in pkt:
        arp_pkt = ARPPacket.arp_parse(msg.data)
        if arp_pkt.opcode == ARP_REQUEST:
            for s in self.session.values():
                if s.dpid == datapath.id and s.ofport == in_port and (s.src_ip == arp_pkt.dst_ip):
                    ans = ARPPacket.arp_packet(ARP_REPLY, s.src_mac, s.src_ip, arp_pkt.src_mac, arp_pkt.src_ip)
                    actions = [parser.OFPActionOutput(in_port)]
                    out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, in_port=ofproto.OFPP_CONTROLLER, actions=actions, data=ans)
                    datapath.send_msg(out)
                    return
        return
    if ipv4.ipv4 not in pkt or udp.udp not in pkt:
        return
    udp_hdr = pkt.get_protocols(udp.udp)[0]
    if udp_hdr.dst_port != BFD_CONTROL_UDP_PORT:
        return
    self.recv_bfd_pkt(datapath, in_port, msg.data)