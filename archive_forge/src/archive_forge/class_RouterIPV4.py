import contextlib
import greenlet
import socket
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.lib import hub
from os_ken.lib import mac as mac_lib
from os_ken.lib.packet import arp
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.ofproto import ether
from os_ken.ofproto import ofproto_v1_2
from os_ken.services.protocols.vrrp import api as vrrp_api
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
class RouterIPV4(RouterBase):

    def _garp_packet(self, ip_address):
        src_mac = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        e = ethernet.ethernet(mac_lib.BROADCAST_STR, src_mac, ether.ETH_TYPE_ARP)
        a = arp.arp_ip(arp.ARP_REQUEST, src_mac, ip_address, mac_lib.DONTCARE_STR, ip_address)
        p = packet.Packet()
        p.add_protocol(e)
        utils.may_add_vlan(p, self.interface.vlan_id)
        p.add_protocol(a)
        p.serialize()
        return p

    def __init__(self, *args, **kwargs):
        super(RouterIPV4, self).__init__(*args, **kwargs)
        assert not self.config.is_ipv6
        self.garp_packets = [self._garp_packet(ip_address) for ip_address in self.config.ip_addresses]

    def _send_garp(self):
        self.logger.debug('_send_garp')
        for garp_packet in self.garp_packets:
            self._transmit(garp_packet.data)

    def _arp_reply_packet(self, arp_req_sha, arp_req_spa, arp_req_tpa):
        if not (arp_req_tpa in self.config.ip_addresses or arp_req_tpa == self.config.primary_ip_address):
            return None
        src_mac = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        e = ethernet.ethernet(arp_req_sha, src_mac, ether.ETH_TYPE_ARP)
        a = arp.arp_ip(arp.ARP_REPLY, src_mac, arp_req_tpa, arp_req_sha, arp_req_spa)
        p = packet.Packet()
        p.add_protocol(e)
        utils.may_add_vlan(p, self.interface.vlan_id)
        p.add_protocol(a)
        p.serialize()
        self._transmit(p.data)

    def _arp_process(self, data):
        dst_mac = vrrp.vrrp_ipv4_src_mac_address(self.config.vrid)
        arp_sha = None
        arp_spa = None
        arp_tpa = None
        p = packet.Packet(data)
        for proto in p.protocols:
            if isinstance(proto, ethernet.ethernet):
                if proto.dst not in (mac_lib.BROADCAST_STR, dst_mac):
                    return None
                ethertype = proto.ethertype
                if not (self.interface.vlan_id is None and ethertype == ether.ETH_TYPE_ARP or (self.interface.vlan_id is not None and ethertype == ether.ETH_TYPE_8021Q)):
                    return None
            elif isinstance(proto, vlan.vlan):
                if proto.vid != self.interface.vlan_id or proto.ethertype != ether.ETH_TYPE_ARP:
                    return None
            elif isinstance(proto, arp.arp):
                if proto.hwtype != arp.ARP_HW_TYPE_ETHERNET or proto.proto != ether.ETH_TYPE_IP or proto.hlen != 6 or (proto.plen != 4) or (proto.opcode != arp.ARP_REQUEST) or (proto.dst_mac != dst_mac):
                    return None
                arp_sha = proto.src_mac
                arp_spa = proto.src_ip
                arp_tpa = proto.dst_ip
                break
        if arp_sha is None or arp_spa is None or arp_tpa is None:
            self.logger.debug('malformed arp request? arp_sha %s arp_spa %s', arp_sha, arp_spa)
            return None
        self._arp_reply_packet(arp_sha, arp_spa, arp_tpa)