import contextlib
import socket
import struct
from os_ken.controller import handler
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.packet import arp
from os_ken.lib.packet import vrrp
from os_ken.services.protocols.vrrp import monitor
from os_ken.services.protocols.vrrp import event as vrrp_event
from os_ken.services.protocols.vrrp import utils
@monitor.VRRPInterfaceMonitor.register(vrrp_event.VRRPInterfaceNetworkDevice)
class VRRPInterfaceMonitorNetworkDevice(monitor.VRRPInterfaceMonitor):
    """
    This module uses raw socket so that privilege(CAP_NET_ADMIN capability)
    is required.
    """

    def __init__(self, *args, **kwargs):
        super(VRRPInterfaceMonitorNetworkDevice, self).__init__(*args, **kwargs)
        self.__is_active = True
        config = self.config
        if config.is_ipv6:
            family = socket.AF_INET6
            ether_type = ether.ETH_TYPE_IPV6
            mac_address = vrrp.vrrp_ipv6_src_mac_address(config.vrid)
        else:
            family = socket.AF_INET
            ether_type = ether.ETH_TYPE_IP
            mac_address = vrrp.vrrp_ipv4_src_mac_address(config.vrid)
        self.ip_socket = socket.socket(family, socket.SOCK_RAW, inet.IPPROTO_VRRP)
        self.packet_socket = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.htons(ether_type))
        self.packet_socket.bind((self.interface.device_name, ether_type, socket.PACKET_MULTICAST, arp.ARP_HW_TYPE_ETHERNET, addrconv.mac.text_to_bin(mac_address)))
        self.ifindex = if_nametoindex(self.interface.device_name)

    def start(self):
        packet_socket = self.packet_socket
        packet_socket.setblocking(0)
        with hub.Timeout(0.1, False):
            while True:
                try:
                    packet_socket.recv(1500)
                except socket.error:
                    break
        packet_socket.setblocking(1)
        self._join_multicast_membership(True)
        self._join_vrrp_group(True)
        super(VRRPInterfaceMonitorNetworkDevice, self).start()
        self.threads.append(hub.spawn(self._recv_loop))

    def stop(self):
        self.__is_active = False
        super(VRRPInterfaceMonitorNetworkDevice, self).stop()

    def _join_multicast_membership(self, join_leave):
        config = self.config
        if config.is_ipv6:
            mac_address = vrrp.vrrp_ipv6_src_mac_address(config.vrid)
        else:
            mac_address = vrrp.vrrp_ipv4_src_mac_address(config.vrid)
        if join_leave:
            add_drop = PACKET_ADD_MEMBERSHIP
        else:
            add_drop = PACKET_DROP_MEMBERSHIP
        packet_mreq = struct.pack('IHH8s', self.ifindex, PACKET_MR_MULTICAST, 6, addrconv.mac.text_to_bin(mac_address))
        self.packet_socket.setsockopt(SOL_PACKET, add_drop, packet_mreq)

    def _join_vrrp_group(self, join_leave):
        if join_leave:
            join_leave = MCAST_JOIN_GROUP
        else:
            join_leave = MCAST_LEAVE_GROUP
        group_req = struct.pack('I', self.ifindex)
        group_req += b'\x00' * (struct.calcsize('P') - struct.calcsize('I'))
        if self.config.is_ipv6:
            family = socket.IPPROTO_IPV6
            sockaddr = struct.pack('H', socket.AF_INET6)
            sockaddr += struct.pack('!H', 0)
            sockaddr += struct.pack('!I', 0)
            sockaddr += addrconv.ipv6.text_to_bin(vrrp.VRRP_IPV6_DST_ADDRESS)
            sockaddr += struct.pack('I', 0)
        else:
            family = socket.IPPROTO_IP
            sockaddr = struct.pack('H', socket.AF_INET)
            sockaddr += struct.pack('!H', 0)
            sockaddr += addrconv.ipv4.text_to_bin(vrrp.VRRP_IPV4_DST_ADDRESS)
        sockaddr += b'\x00' * (SS_MAXSIZE - len(sockaddr))
        group_req += sockaddr
        self.ip_socket.setsockopt(family, join_leave, group_req)
        return

    def _recv_loop(self):
        packet_socket = self.packet_socket
        packet_socket.settimeout(1.3)
        try:
            while self.__is_active:
                try:
                    buf = packet_socket.recv(128)
                except socket.timeout:
                    self.logger.debug('timeout')
                    continue
                except:
                    self.logger.error('recv failed')
                    continue
                if len(buf) == 0:
                    self.__is_active = False
                    break
                self.logger.debug('recv buf')
                self._send_vrrp_packet_received(buf)
        finally:
            self._join_vrrp_group(False)
            self._join_multicast_membership(False)

    @handler.set_ev_handler(vrrp_event.EventVRRPTransmitRequest)
    def vrrp_transmit_request_handler(self, ev):
        self.logger.debug('send')
        try:
            self.packet_socket.sendto(ev.data, (self.interface.device_name, 0))
        except:
            self.logger.error('send failed')

    def _initialize(self):
        pass

    def _shutdown(self):
        self.__is_active = False