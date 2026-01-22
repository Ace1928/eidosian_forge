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
def _listen_socket_loop(self, s, conn_handle):
    while True:
        sock, client_address = s.accept()
        client_address, port = self.get_remotename(sock)
        LOG.debug('Connect request received from client for port %s:%s', client_address, port)
        client_name = self.name + '_client@' + client_address
        self._asso_socket_map[client_name] = sock
        self._spawn(client_name, conn_handle, sock)