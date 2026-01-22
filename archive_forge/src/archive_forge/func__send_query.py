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
def _send_query(self):
    """ send a QUERY message periodically."""
    timeout = 60
    ofproto = self._datapath.ofproto
    parser = self._datapath.ofproto_parser
    if ofproto_v1_0.OFP_VERSION == ofproto.OFP_VERSION:
        send_port = ofproto.OFPP_NONE
    else:
        send_port = ofproto.OFPP_ANY
    res_igmp = igmp.igmp(msgtype=igmp.IGMP_TYPE_QUERY, maxresp=igmp.QUERY_RESPONSE_INTERVAL * 10, csum=0, address='0.0.0.0')
    res_ipv4 = ipv4.ipv4(total_length=len(ipv4.ipv4()) + len(res_igmp), proto=inet.IPPROTO_IGMP, ttl=1, src='0.0.0.0', dst=igmp.MULTICAST_IP_ALL_HOST)
    res_ether = ethernet.ethernet(dst=igmp.MULTICAST_MAC_ALL_HOST, src=self._datapath.ports[ofproto.OFPP_LOCAL].hw_addr, ethertype=ether.ETH_TYPE_IP)
    res_pkt = packet.Packet()
    res_pkt.add_protocol(res_ether)
    res_pkt.add_protocol(res_ipv4)
    res_pkt.add_protocol(res_igmp)
    res_pkt.serialize()
    flood = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
    while True:
        for status in self._mcast.values():
            for port in status.keys():
                status[port] = False
        self._do_packet_out(self._datapath, res_pkt.data, send_port, flood)
        hub.sleep(igmp.QUERY_RESPONSE_INTERVAL)
        del_groups = []
        for group, status in self._mcast.items():
            del_ports = []
            actions = []
            for port in status.keys():
                if not status[port]:
                    del_ports.append(port)
                else:
                    actions.append(parser.OFPActionOutput(port))
            if len(actions) and len(del_ports):
                self._set_flow_entry(self._datapath, actions, self.server_port, group)
            if not len(actions):
                self._del_flow_entry(self._datapath, self.server_port, group)
                del_groups.append(group)
            if len(del_ports):
                for port in del_ports:
                    self._del_flow_entry(self._datapath, port, group)
            for port in del_ports:
                del status[port]
        for group in del_groups:
            del self._mcast[group]
        rest_time = timeout - igmp.QUERY_RESPONSE_INTERVAL
        hub.sleep(rest_time)