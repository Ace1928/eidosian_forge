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
class FlexinetPeer(Source, Sink):

    def __init__(self):
        Source.__init__(self, 1)
        Sink.__init__(self)