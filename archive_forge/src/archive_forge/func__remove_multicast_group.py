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
def _remove_multicast_group(self, datapath, outport, dst):
    """remove flow entries about the group and send a LEAVE message
        if exists."""
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    dpid = datapath.id
    self._send_event(EventMulticastGroupStateChanged(MG_GROUP_REMOVED, dst, outport, []))
    self._del_flow_entry(datapath, outport, dst)
    for port in self._to_hosts[dpid][dst]['ports']:
        self._del_flow_entry(datapath, port, dst)
    leave = self._to_hosts[dpid][dst]['leave']
    if leave:
        if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            in_port = leave.in_port
        else:
            in_port = leave.match['in_port']
        actions = [parser.OFPActionOutput(outport)]
        self._do_packet_out(datapath, leave.data, in_port, actions)