import abc
from collections import OrderedDict
import logging
import socket
import time
import traceback
import weakref
import netaddr
from os_ken.lib import hub
from os_ken.lib import sockopt
from os_ken.lib import ip
from os_ken.lib.hub import Timeout
from os_ken.lib.packet.bgp import RF_IPv4_UC
from os_ken.lib.packet.bgp import RF_IPv6_UC
from os_ken.lib.packet.bgp import RF_IPv4_VPN
from os_ken.lib.packet.bgp import RF_IPv6_VPN
from os_ken.lib.packet.bgp import RF_L2_EVPN
from os_ken.lib.packet.bgp import RF_IPv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_IPv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv4_FLOWSPEC
from os_ken.lib.packet.bgp import RF_VPNv6_FLOWSPEC
from os_ken.lib.packet.bgp import RF_L2VPN_FLOWSPEC
from os_ken.lib.packet.bgp import RF_RTC_UC
from os_ken.services.protocols.bgp.utils.circlist import CircularListType
from os_ken.services.protocols.bgp.utils.evtlet import LoopingCall
def _listen_tcp(self, loc_addr, conn_handle):
    """Creates a TCP server socket which listens on `port` number.

        For each connection `server_factory` starts a new protocol.
        """
    info = socket.getaddrinfo(loc_addr[0], loc_addr[1], socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
    listen_sockets = {}
    for res in info:
        af, socktype, proto, _, sa = res
        sock = None
        try:
            sock = socket.socket(af, socktype, proto)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if af == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            sock.bind(sa)
            sock.listen(50)
            listen_sockets[sa] = sock
        except socket.error as e:
            LOG.error('Error creating socket: %s', e)
            if sock:
                sock.close()
    count = 0
    server = None
    for sa in listen_sockets:
        name = self.name + '_server@' + str(sa[0])
        self._asso_socket_map[name] = listen_sockets[sa]
        if count == 0:
            import eventlet
            server = eventlet.spawn(self._listen_socket_loop, listen_sockets[sa], conn_handle)
            self._child_thread_map[name] = server
            count += 1
        else:
            server = self._spawn(name, self._listen_socket_loop, listen_sockets[sa], conn_handle)
    return (server, listen_sockets)