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
class IgmpLib(app_manager.OSKenApp):
    """IGMP snooping library."""

    def __init__(self):
        """initialization."""
        super(IgmpLib, self).__init__()
        self.name = 'igmplib'
        self._querier = IgmpQuerier()
        self._snooper = IgmpSnooper(self.send_event_to_observers)

    def set_querier_mode(self, dpid, server_port):
        """set a datapath id and server port number to the instance
        of IgmpQuerier.

        ============ ==================================================
        Attribute    Description
        ============ ==================================================
        dpid         the datapath id that will operate as a querier.
        server_port  the port number linked to the multicasting server.
        ============ ==================================================
        """
        self._querier.set_querier_mode(dpid, server_port)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, evt):
        """PacketIn event handler. when the received packet was IGMP,
        proceed it. otherwise, send a event."""
        msg = evt.msg
        dpid = msg.datapath.id
        req_pkt = packet.Packet(msg.data)
        req_igmp = req_pkt.get_protocol(igmp.igmp)
        if req_igmp:
            if self._querier.dpid == dpid:
                self._querier.packet_in_handler(req_igmp, msg)
            else:
                self._snooper.packet_in_handler(req_pkt, req_igmp, msg)
        else:
            self.send_event_to_observers(EventPacketIn(msg))

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, evt):
        """StateChange event handler."""
        datapath = evt.datapath
        assert datapath is not None
        if datapath.id == self._querier.dpid:
            if evt.state == MAIN_DISPATCHER:
                self._querier.start_loop(datapath)
            elif evt.state == DEAD_DISPATCHER:
                self._querier.stop_loop()