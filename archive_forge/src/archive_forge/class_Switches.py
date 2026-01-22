import logging
import struct
import time
from os_ken import cfg
from collections import defaultdict
from os_ken.topology import event
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.controller.handler import MAIN_DISPATCHER, DEAD_DISPATCHER
from os_ken.exception import OSKenException
from os_ken.lib import addrconv, hub
from os_ken.lib.mac import DONTCARE_STR
from os_ken.lib.dpid import dpid_to_str, str_to_dpid
from os_ken.lib.port_no import port_no_to_str
from os_ken.lib.packet import packet, ethernet
from os_ken.lib.packet import lldp, ether_types
from os_ken.ofproto.ether import ETH_TYPE_LLDP
from os_ken.ofproto.ether import ETH_TYPE_CFM
from os_ken.ofproto import nx_match
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
class Switches(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_0.OFP_VERSION, ofproto_v1_2.OFP_VERSION, ofproto_v1_3.OFP_VERSION, ofproto_v1_4.OFP_VERSION]
    _EVENTS = [event.EventSwitchEnter, event.EventSwitchLeave, event.EventSwitchReconnected, event.EventPortAdd, event.EventPortDelete, event.EventPortModify, event.EventLinkAdd, event.EventLinkDelete, event.EventHostAdd]
    DEFAULT_TTL = 120
    LLDP_PACKET_LEN = len(LLDPPacket.lldp_packet(0, 0, DONTCARE_STR, 0))
    LLDP_SEND_GUARD = 0.05
    LLDP_SEND_PERIOD_PER_PORT = 0.9
    TIMEOUT_CHECK_PERIOD = 5.0
    LINK_TIMEOUT = TIMEOUT_CHECK_PERIOD * 2

    def __init__(self, *args, **kwargs):
        super(Switches, self).__init__(*args, **kwargs)
        self.name = 'switches'
        self.dps = {}
        self.port_state = {}
        self.ports = PortDataState()
        self.links = LinkState()
        self.hosts = HostState()
        self.is_active = True
        self.link_discovery = self.CONF.observe_links
        if self.link_discovery:
            self.install_flow = self.CONF.install_lldp_flow
            self.explicit_drop = self.CONF.explicit_drop
            self.lldp_event = hub.Event()
            self.link_event = hub.Event()
            self.threads.append(hub.spawn(self.lldp_loop))
            self.threads.append(hub.spawn(self.link_loop))

    def close(self):
        self.is_active = False
        if self.link_discovery:
            self.lldp_event.set()
            self.link_event.set()
            hub.joinall(self.threads)

    def _register(self, dp):
        assert dp.id is not None
        self.dps[dp.id] = dp
        if dp.id not in self.port_state:
            self.port_state[dp.id] = PortState()
            for port in dp.ports.values():
                self.port_state[dp.id].add(port.port_no, port)

    def _unregister(self, dp):
        if dp.id in self.dps:
            if self.dps[dp.id] == dp:
                del self.dps[dp.id]
                del self.port_state[dp.id]

    def _get_switch(self, dpid):
        if dpid in self.dps:
            switch = Switch(self.dps[dpid])
            for ofpport in self.port_state[dpid].values():
                switch.add_port(ofpport)
            return switch

    def _get_port(self, dpid, port_no):
        switch = self._get_switch(dpid)
        if switch:
            for p in switch.ports:
                if p.port_no == port_no:
                    return p

    def _port_added(self, port):
        lldp_data = LLDPPacket.lldp_packet(port.dpid, port.port_no, port.hw_addr, self.DEFAULT_TTL)
        self.ports.add_port(port, lldp_data)

    def _link_down(self, port):
        try:
            dsts, rev_link_dsts = self.links.port_deleted(port)
        except KeyError:
            return
        for dst in dsts:
            link = Link(port, dst)
            self.send_event_to_observers(event.EventLinkDelete(link))
        for rev_link_dst in rev_link_dsts:
            rev_link = Link(rev_link_dst, port)
            self.send_event_to_observers(event.EventLinkDelete(rev_link))
            self.ports.move_front(rev_link_dst)

    def _is_edge_port(self, port):
        for link in self.links:
            if port == link.src or port == link.dst:
                return False
        return True

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        dp = ev.datapath
        assert dp is not None
        LOG.debug(dp)
        if ev.state == MAIN_DISPATCHER:
            dp_multiple_conns = False
            if dp.id in self.dps:
                LOG.warning('Multiple connections from %s', dpid_to_str(dp.id))
                dp_multiple_conns = True
                self.dps[dp.id].close()
            self._register(dp)
            switch = self._get_switch(dp.id)
            LOG.debug('register %s', switch)
            if not dp_multiple_conns:
                self.send_event_to_observers(event.EventSwitchEnter(switch))
            else:
                evt = event.EventSwitchReconnected(switch)
                self.send_event_to_observers(evt)
            if not self.link_discovery:
                return
            if self.install_flow:
                ofproto = dp.ofproto
                ofproto_parser = dp.ofproto_parser
                if ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
                    rule = nx_match.ClsRule()
                    rule.set_dl_dst(addrconv.mac.text_to_bin(lldp.LLDP_MAC_NEAREST_BRIDGE))
                    rule.set_dl_type(ETH_TYPE_LLDP)
                    actions = [ofproto_parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, self.LLDP_PACKET_LEN)]
                    dp.send_flow_mod(rule=rule, cookie=0, command=ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0, actions=actions, priority=65535)
                elif ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
                    match = ofproto_parser.OFPMatch(eth_type=ETH_TYPE_LLDP, eth_dst=lldp.LLDP_MAC_NEAREST_BRIDGE)
                    parser = ofproto_parser
                    actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
                    inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
                    mod = parser.OFPFlowMod(datapath=dp, match=match, idle_timeout=0, hard_timeout=0, instructions=inst, priority=65535)
                    dp.send_msg(mod)
                else:
                    LOG.error('cannot install flow. unsupported version. %x', dp.ofproto.OFP_VERSION)
            if not dp_multiple_conns:
                for port in switch.ports:
                    if not port.is_reserved():
                        self._port_added(port)
            self.lldp_event.set()
        elif ev.state == DEAD_DISPATCHER:
            if dp.id is None:
                return
            switch = self._get_switch(dp.id)
            if switch:
                if switch.dp is dp:
                    self._unregister(dp)
                    LOG.debug('unregister %s', switch)
                    evt = event.EventSwitchLeave(switch)
                    self.send_event_to_observers(evt)
                    if not self.link_discovery:
                        return
                    for port in switch.ports:
                        if not port.is_reserved():
                            self.ports.del_port(port)
                            self._link_down(port)
                    self.lldp_event.set()

    @set_ev_cls(ofp_event.EventOFPPortStatus, MAIN_DISPATCHER)
    def port_status_handler(self, ev):
        msg = ev.msg
        reason = msg.reason
        dp = msg.datapath
        ofpport = msg.desc
        if reason == dp.ofproto.OFPPR_ADD:
            self.port_state[dp.id].add(ofpport.port_no, ofpport)
            self.send_event_to_observers(event.EventPortAdd(Port(dp.id, dp.ofproto, ofpport)))
            if not self.link_discovery:
                return
            port = self._get_port(dp.id, ofpport.port_no)
            if port and (not port.is_reserved()):
                self._port_added(port)
                self.lldp_event.set()
        elif reason == dp.ofproto.OFPPR_DELETE:
            self.send_event_to_observers(event.EventPortDelete(Port(dp.id, dp.ofproto, ofpport)))
            if not self.link_discovery:
                return
            port = self._get_port(dp.id, ofpport.port_no)
            if port and (not port.is_reserved()):
                self.ports.del_port(port)
                self._link_down(port)
                self.lldp_event.set()
            self.port_state[dp.id].remove(ofpport.port_no)
        else:
            assert reason == dp.ofproto.OFPPR_MODIFY
            self.port_state[dp.id].modify(ofpport.port_no, ofpport)
            self.send_event_to_observers(event.EventPortModify(Port(dp.id, dp.ofproto, ofpport)))
            if not self.link_discovery:
                return
            port = self._get_port(dp.id, ofpport.port_no)
            if port and (not port.is_reserved()):
                if self.ports.set_down(port):
                    self._link_down(port)
                self.lldp_event.set()

    @staticmethod
    def _drop_packet(msg):
        buffer_id = msg.buffer_id
        if buffer_id == msg.datapath.ofproto.OFP_NO_BUFFER:
            return
        dp = msg.datapath
        if dp.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            dp.send_packet_out(buffer_id, msg.in_port, [])
        elif dp.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            dp.send_packet_out(buffer_id, msg.match['in_port'], [])
        else:
            LOG.error('cannot drop_packet. unsupported version. %x', dp.ofproto.OFP_VERSION)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def lldp_packet_in_handler(self, ev):
        if not self.link_discovery:
            return
        msg = ev.msg
        try:
            src_dpid, src_port_no = LLDPPacket.lldp_parse(msg.data)
        except LLDPPacket.LLDPUnknownFormat:
            return
        dst_dpid = msg.datapath.id
        if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            dst_port_no = msg.in_port
        elif msg.datapath.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            dst_port_no = msg.match['in_port']
        else:
            LOG.error('cannot accept LLDP. unsupported version. %x', msg.datapath.ofproto.OFP_VERSION)
        src = self._get_port(src_dpid, src_port_no)
        if not src or src.dpid == dst_dpid:
            return
        try:
            self.ports.lldp_received(src)
        except KeyError:
            pass
        dst = self._get_port(dst_dpid, dst_port_no)
        if not dst:
            return
        link = Link(src, dst)
        if link not in self.links:
            self.send_event_to_observers(event.EventLinkAdd(link))
            host_to_del = []
            for host in self.hosts.values():
                if not self._is_edge_port(host.port):
                    host_to_del.append(host.mac)
            for host_mac in host_to_del:
                del self.hosts[host_mac]
        if not self.links.update_link(src, dst):
            self.ports.move_front(dst)
            self.lldp_event.set()
        if self.explicit_drop:
            self._drop_packet(msg)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def host_discovery_packet_in_handler(self, ev):
        msg = ev.msg
        eth, pkt_type, pkt_data = ethernet.ethernet.parser(msg.data)
        if eth.ethertype in (ETH_TYPE_LLDP, ETH_TYPE_CFM):
            return
        datapath = msg.datapath
        dpid = datapath.id
        port_no = -1
        if msg.datapath.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            port_no = msg.in_port
        else:
            port_no = msg.match['in_port']
        port = self._get_port(dpid, port_no)
        if not port:
            return
        if not self._is_edge_port(port):
            return
        host_mac = eth.src
        host = Host(host_mac, port)
        if host_mac not in self.hosts:
            self.hosts.add(host)
            ev = event.EventHostAdd(host)
            self.send_event_to_observers(ev)
        elif self.hosts[host_mac].port != port:
            ev = event.EventHostMove(src=self.hosts[host_mac], dst=host)
            self.hosts[host_mac] = host
            self.send_event_to_observers(ev)
        if eth.ethertype == ether_types.ETH_TYPE_ARP:
            arp_pkt, _, _ = pkt_type.parser(pkt_data)
            self.hosts.update_ip(host, ip_v4=arp_pkt.src_ip)
        elif eth.ethertype == ether_types.ETH_TYPE_IP:
            ipv4_pkt, _, _ = pkt_type.parser(pkt_data)
            self.hosts.update_ip(host, ip_v4=ipv4_pkt.src)
        elif eth.ethertype == ether_types.ETH_TYPE_IPV6:
            ipv6_pkt, _, _ = pkt_type.parser(pkt_data)
            self.hosts.update_ip(host, ip_v6=ipv6_pkt.src)

    def send_lldp_packet(self, port):
        try:
            port_data = self.ports.lldp_sent(port)
        except KeyError:
            return
        if port_data.is_down:
            return
        dp = self.dps.get(port.dpid, None)
        if dp is None:
            return
        if dp.ofproto.OFP_VERSION == ofproto_v1_0.OFP_VERSION:
            actions = [dp.ofproto_parser.OFPActionOutput(port.port_no)]
            dp.send_packet_out(actions=actions, data=port_data.lldp_data)
        elif dp.ofproto.OFP_VERSION >= ofproto_v1_2.OFP_VERSION:
            actions = [dp.ofproto_parser.OFPActionOutput(port.port_no)]
            out = dp.ofproto_parser.OFPPacketOut(datapath=dp, in_port=dp.ofproto.OFPP_CONTROLLER, buffer_id=dp.ofproto.OFP_NO_BUFFER, actions=actions, data=port_data.lldp_data)
            dp.send_msg(out)
        else:
            LOG.error('cannot send lldp packet. unsupported version. %x', dp.ofproto.OFP_VERSION)

    def lldp_loop(self):
        while self.is_active:
            self.lldp_event.clear()
            now = time.time()
            timeout = None
            ports_now = []
            ports = []
            for key, data in self.ports.items():
                if data.timestamp is None:
                    ports_now.append(key)
                    continue
                expire = data.timestamp + self.LLDP_SEND_PERIOD_PER_PORT
                if expire <= now:
                    ports.append(key)
                    continue
                timeout = expire - now
                break
            for port in ports_now:
                self.send_lldp_packet(port)
            for port in ports:
                self.send_lldp_packet(port)
                hub.sleep(self.LLDP_SEND_GUARD)
            if timeout is not None and ports:
                timeout = 0
            self.lldp_event.wait(timeout=timeout)

    def link_loop(self):
        while self.is_active:
            self.link_event.clear()
            now = time.time()
            deleted = []
            for link, timestamp in self.links.items():
                if timestamp + self.LINK_TIMEOUT < now:
                    deleted.append(link)
            for link in deleted:
                self.links.link_down(link)
                self.send_event_to_observers(event.EventLinkDelete(link))
                dst = link.dst
                rev_link = Link(dst, link.src)
                if rev_link not in deleted:
                    expire = now - self.LINK_TIMEOUT
                    self.links.rev_link_set_timestamp(rev_link, expire)
                    if dst in self.ports:
                        self.ports.move_front(dst)
                        self.lldp_event.set()
            self.link_event.wait(timeout=self.TIMEOUT_CHECK_PERIOD)

    @set_ev_cls(event.EventSwitchRequest)
    def switch_request_handler(self, req):
        dpid = req.dpid
        switches = []
        if dpid is None:
            for dp in self.dps.values():
                switches.append(self._get_switch(dp.id))
        elif dpid in self.dps:
            switches.append(self._get_switch(dpid))
        rep = event.EventSwitchReply(req.src, switches)
        self.reply_to_request(req, rep)

    @set_ev_cls(event.EventLinkRequest)
    def link_request_handler(self, req):
        dpid = req.dpid
        if dpid is None:
            links = self.links
        else:
            links = [link for link in self.links if link.src.dpid == dpid]
        rep = event.EventLinkReply(req.src, dpid, links)
        self.reply_to_request(req, rep)

    @set_ev_cls(event.EventHostRequest)
    def host_request_handler(self, req):
        dpid = req.dpid
        hosts = []
        if dpid is None:
            for mac in self.hosts:
                hosts.append(self.hosts[mac])
        else:
            hosts = self.hosts.get_by_dpid(dpid)
        rep = event.EventHostReply(req.src, dpid, hosts)
        self.reply_to_request(req, rep)