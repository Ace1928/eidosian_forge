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
class IgmpSnooper(IgmpBase):
    """IGMP snooping class library."""

    def __init__(self, send_event):
        """initialization."""
        super(IgmpSnooper, self).__init__()
        self.name = 'IgmpSnooper'
        self.logger = logging.getLogger(self.name)
        self._send_event = send_event
        self._to_querier = {}
        self._to_hosts = {}
        self._set_logger()

    def packet_in_handler(self, req_pkt, req_igmp, msg):
        """the process when the snooper received IGMP."""
        dpid = msg.datapath.id
        ofproto = msg.datapath.ofproto
        if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            in_port = msg.in_port
        else:
            in_port = msg.match['in_port']
        log = 'SW=%s PORT=%d IGMP received. ' % (dpid_to_str(dpid), in_port)
        self.logger.debug(str(req_igmp))
        if igmp.IGMP_TYPE_QUERY == req_igmp.msgtype:
            self.logger.info(log + '[QUERY]')
            req_ipv4, = req_pkt.get_protocols(ipv4.ipv4)
            req_eth, = req_pkt.get_protocols(ethernet.ethernet)
            self._do_query(req_igmp, req_ipv4, req_eth, in_port, msg)
        elif igmp.IGMP_TYPE_REPORT_V1 == req_igmp.msgtype or igmp.IGMP_TYPE_REPORT_V2 == req_igmp.msgtype:
            self.logger.info(log + '[REPORT]')
            self._do_report(req_igmp, in_port, msg)
        elif igmp.IGMP_TYPE_LEAVE == req_igmp.msgtype:
            self.logger.info(log + '[LEAVE]')
            self._do_leave(req_igmp, in_port, msg)
        elif igmp.IGMP_TYPE_REPORT_V3 == req_igmp.msgtype:
            self.logger.info(log + 'V3 is not supported yet.')
            self._do_flood(in_port, msg)
        else:
            self.logger.info(log + '[unknown type:%d]', req_igmp.msgtype)
            self._do_flood(in_port, msg)

    def _do_query(self, query, iph, eth, in_port, msg):
        """the process when the snooper received a QUERY message."""
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        self._to_querier[dpid] = {'port': in_port, 'ip': iph.src, 'mac': eth.src}
        timeout = igmp.QUERY_RESPONSE_INTERVAL
        if query.maxresp:
            timeout = query.maxresp / 10
        self._to_hosts.setdefault(dpid, {})
        if query.address == '0.0.0.0':
            for group in self._to_hosts[dpid].values():
                group['replied'] = False
                group['leave'] = None
        else:
            group = self._to_hosts[dpid].get(query.address)
            if group:
                group['replied'] = False
                group['leave'] = None
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        self._do_packet_out(datapath, msg.data, in_port, actions)
        hub.spawn(self._do_timeout_for_query, timeout, datapath)

    def _do_report(self, report, in_port, msg):
        """the process when the snooper received a REPORT message."""
        datapath = msg.datapath
        dpid = datapath.id
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            size = 65535
        else:
            size = ofproto.OFPCML_MAX
        outport = None
        value = self._to_querier.get(dpid)
        if value:
            outport = value['port']
        self._to_hosts.setdefault(dpid, {})
        if not self._to_hosts[dpid].get(report.address):
            self._send_event(EventMulticastGroupStateChanged(MG_GROUP_ADDED, report.address, outport, []))
            self._to_hosts[dpid].setdefault(report.address, {'replied': False, 'leave': None, 'ports': {}})
        if not self._to_hosts[dpid][report.address]['ports'].get(in_port):
            self._to_hosts[dpid][report.address]['ports'][in_port] = {'out': False, 'in': False}
            self._set_flow_entry(datapath, [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, size)], in_port, report.address)
        if not self._to_hosts[dpid][report.address]['ports'][in_port]['out']:
            self._to_hosts[dpid][report.address]['ports'][in_port]['out'] = True
        if not outport:
            self.logger.info('no querier exists.')
            return
        if not self._to_hosts[dpid][report.address]['ports'][in_port]['in']:
            actions = []
            ports = []
            for port in self._to_hosts[dpid][report.address]['ports']:
                actions.append(parser.OFPActionOutput(port))
                ports.append(port)
            self._send_event(EventMulticastGroupStateChanged(MG_MEMBER_CHANGED, report.address, outport, ports))
            self._set_flow_entry(datapath, actions, outport, report.address)
            self._to_hosts[dpid][report.address]['ports'][in_port]['in'] = True
        if not self._to_hosts[dpid][report.address]['replied']:
            actions = [parser.OFPActionOutput(outport, size)]
            self._do_packet_out(datapath, msg.data, in_port, actions)
            self._to_hosts[dpid][report.address]['replied'] = True

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

    def _do_flood(self, in_port, msg):
        """the process when the snooper received a message of the
        outside for processing. """
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        actions = [parser.OFPActionOutput(ofproto.OFPP_FLOOD)]
        self._do_packet_out(datapath, msg.data, in_port, actions)

    def _do_timeout_for_query(self, timeout, datapath):
        """the process when the QUERY from the querier timeout expired."""
        dpid = datapath.id
        hub.sleep(timeout)
        outport = self._to_querier[dpid]['port']
        remove_dsts = []
        for dst in self._to_hosts[dpid]:
            if not self._to_hosts[dpid][dst]['replied']:
                self._remove_multicast_group(datapath, outport, dst)
                remove_dsts.append(dst)
        for dst in remove_dsts:
            del self._to_hosts[dpid][dst]

    def _do_timeout_for_leave(self, timeout, datapath, dst, in_port):
        """the process when the QUERY from the switch timeout expired."""
        parser = datapath.ofproto_parser
        dpid = datapath.id
        hub.sleep(timeout)
        outport = self._to_querier[dpid]['port']
        if self._to_hosts[dpid][dst]['ports'][in_port]['out']:
            return
        del self._to_hosts[dpid][dst]['ports'][in_port]
        self._del_flow_entry(datapath, in_port, dst)
        actions = []
        ports = []
        for port in self._to_hosts[dpid][dst]['ports']:
            actions.append(parser.OFPActionOutput(port))
            ports.append(port)
        if len(actions):
            self._send_event(EventMulticastGroupStateChanged(MG_MEMBER_CHANGED, dst, outport, ports))
            self._set_flow_entry(datapath, actions, outport, dst)
            self._to_hosts[dpid][dst]['leave'] = None
        else:
            self._remove_multicast_group(datapath, outport, dst)
            del self._to_hosts[dpid][dst]

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

    def _set_logger(self):
        """change log format."""
        self.logger.propagate = False
        hdl = logging.StreamHandler()
        fmt_str = '[snoop][%(levelname)s] %(message)s'
        hdl.setFormatter(logging.Formatter(fmt_str))
        self.logger.addHandler(hdl)