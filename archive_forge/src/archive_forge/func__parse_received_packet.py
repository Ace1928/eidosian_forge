from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.lib.packet import packet
from os_ken.lib.packet import vlan
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import event as vrrp_event
def _parse_received_packet(self, packet_data):
    packet_ = packet.Packet(packet_data)
    protocols = packet_.protocols
    if len(protocols) < 2:
        self.logger.debug('len(protocols) %d', len(protocols))
        return
    vlan_vid = self.interface.vlan_id
    may_vlan = protocols[1]
    if (vlan_vid is not None) != isinstance(may_vlan, vlan.vlan):
        self.logger.debug('vlan_vid: %s %s', vlan_vid, type(may_vlan))
        return
    if vlan_vid is not None and vlan_vid != may_vlan.vid:
        self.logger.debug('vlan_vid: %s vlan %s', vlan_vid, type(may_vlan))
        return
    may_ip, may_vrrp = vrrp.vrrp.get_payload(packet_)
    if not may_ip or not may_vrrp:
        return
    if not vrrp.vrrp.is_valid_ttl(may_ip):
        self.logger.debug('valid_ttl')
        return
    if may_vrrp.version != self.config.version:
        self.logger.debug('vrrp version %d %d', may_vrrp.version, self.config.version)
        return
    if not may_vrrp.is_valid():
        self.logger.debug('valid vrrp')
        return
    offset = 0
    for proto in packet_.protocols:
        if proto == may_vrrp:
            break
        offset += len(proto)
    if not may_vrrp.checksum_ok(may_ip, packet_.data[offset:offset + len(may_vrrp)]):
        self.logger.debug('bad checksum')
        return
    if may_vrrp.vrid != self.config.vrid:
        self.logger.debug('vrid %d %d', may_vrrp.vrid, self.config.vrid)
        return
    if may_vrrp.is_ipv6 != self.config.is_ipv6:
        self.logger.debug('is_ipv6 %s %s', may_vrrp.is_ipv6, self.config.is_ipv6)
        return
    if may_vrrp.priority == 0:
        self.statistics.rx_vrrp_zero_prio_packets += 1
    vrrp_received = vrrp_event.EventVRRPReceived(self.interface, packet_)
    self.send_event(self.router_name, vrrp_received)
    return True