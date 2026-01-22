import logging
import struct
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import ofp_event
from os_ken.controller.handler import DEAD_DISPATCHER
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib import addrconv
from os_ken.lib import hub
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import packet
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import ipv4
from os_ken.lib.packet import igmp
class EventMulticastGroupStateChanged(event.EventBase):
    """a event class that notifies the changes of the statuses of the
    multicast groups."""

    def __init__(self, reason, address, src, dsts):
        """
        ========= =====================================================
        Attribute Description
        ========= =====================================================
        reason    why the event occurs. use one of MG_*.
        address   a multicast group address.
        src       a port number in which a querier exists.
        dsts      a list of port numbers in which the members exist.
        ========= =====================================================
        """
        super(EventMulticastGroupStateChanged, self).__init__()
        self.reason = reason
        self.address = address
        self.src = src
        self.dsts = dsts