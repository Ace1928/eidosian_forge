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
def _do_leave(self, leave, in_port, msg):
    """the process when the snooper received a LEAVE message."""
    datapath = msg.datapath
    dpid = datapath.id
    ofproto = datapath.ofproto
    parser = datapath.ofproto_parser
    if not self._to_querier.get(dpid):
        self.logger.info('no querier exists.')
        return
    self._to_hosts.setdefault(dpid, {})
    self._to_hosts[dpid].setdefault(leave.address, {'replied': False, 'leave': None, 'ports': {}})
    self._to_hosts[dpid][leave.address]['leave'] = msg
    self._to_hosts[dpid][leave.address]['ports'][in_port] = {'out': False, 'in': False}
    timeout = igmp.LAST_MEMBER_QUERY_INTERVAL
    res_igmp = igmp.igmp(msgtype=igmp.IGMP_TYPE_QUERY, maxresp=timeout * 10, csum=0, address=leave.address)
    res_ipv4 = ipv4.ipv4(total_length=len(ipv4.ipv4()) + len(res_igmp), proto=inet.IPPROTO_IGMP, ttl=1, src=self._to_querier[dpid]['ip'], dst=igmp.MULTICAST_IP_ALL_HOST)
    res_ether = ethernet.ethernet(dst=igmp.MULTICAST_MAC_ALL_HOST, src=self._to_querier[dpid]['mac'], ethertype=ether.ETH_TYPE_IP)
    res_pkt = packet.Packet()
    res_pkt.add_protocol(res_ether)
    res_pkt.add_protocol(res_ipv4)
    res_pkt.add_protocol(res_igmp)
    res_pkt.serialize()
    actions = [parser.OFPActionOutput(ofproto.OFPP_IN_PORT)]
    self._do_packet_out(datapath, res_pkt.data, in_port, actions)
    hub.spawn(self._do_timeout_for_leave, timeout, datapath, leave.address, in_port)