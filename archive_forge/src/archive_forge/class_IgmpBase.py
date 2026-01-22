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
class IgmpBase(object):
    """IGMP abstract class library."""

    def __init__(self):
        self._set_flow_func = {ofproto_v1_0.OFP_VERSION: self._set_flow_entry_v1_0, ofproto_v1_2.OFP_VERSION: self._set_flow_entry_v1_2, ofproto_v1_3.OFP_VERSION: self._set_flow_entry_v1_2}
        self._del_flow_func = {ofproto_v1_0.OFP_VERSION: self._del_flow_entry_v1_0, ofproto_v1_2.OFP_VERSION: self._del_flow_entry_v1_2, ofproto_v1_3.OFP_VERSION: self._del_flow_entry_v1_2}

    def _set_flow_entry_v1_0(self, datapath, actions, in_port, dst, src=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(dl_type=ether.ETH_TYPE_IP, in_port=in_port, nw_src=self._ipv4_text_to_int(src), nw_dst=self._ipv4_text_to_int(dst))
        mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0, command=ofproto.OFPFC_ADD, actions=actions)
        datapath.send_msg(mod)

    def _set_flow_entry_v1_2(self, datapath, actions, in_port, dst, src=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=ether.ETH_TYPE_IP, in_port=in_port, ipv4_dst=dst)
        if src is not None:
            match.append_field(ofproto.OXM_OF_IPV4_SRC, src)
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(datapath=datapath, command=ofproto.OFPFC_ADD, priority=65535, match=match, instructions=inst)
        datapath.send_msg(mod)

    def _set_flow_entry(self, datapath, actions, in_port, dst, src=None):
        """set a flow entry."""
        set_flow = self._set_flow_func.get(datapath.ofproto.OFP_VERSION)
        assert set_flow
        set_flow(datapath, actions, in_port, dst, src)

    def _del_flow_entry_v1_0(self, datapath, in_port, dst, src=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(dl_type=ether.ETH_TYPE_IP, in_port=in_port, nw_src=self._ipv4_text_to_int(src), nw_dst=self._ipv4_text_to_int(dst))
        mod = parser.OFPFlowMod(datapath=datapath, match=match, cookie=0, command=ofproto.OFPFC_DELETE)
        datapath.send_msg(mod)

    def _del_flow_entry_v1_2(self, datapath, in_port, dst, src=None):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        match = parser.OFPMatch(eth_type=ether.ETH_TYPE_IP, in_port=in_port, ipv4_dst=dst)
        if src is not None:
            match.append_field(ofproto.OXM_OF_IPV4_SRC, src)
        mod = parser.OFPFlowMod(datapath=datapath, command=ofproto.OFPFC_DELETE, out_port=ofproto.OFPP_ANY, out_group=ofproto.OFPG_ANY, match=match)
        datapath.send_msg(mod)

    def _del_flow_entry(self, datapath, in_port, dst, src=None):
        """remove a flow entry."""
        del_flow = self._del_flow_func.get(datapath.ofproto.OFP_VERSION)
        assert del_flow
        del_flow(datapath, in_port, dst, src)

    def _do_packet_out(self, datapath, data, in_port, actions):
        """send a packet."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=ofproto.OFP_NO_BUFFER, data=data, in_port=in_port, actions=actions)
        datapath.send_msg(out)

    def _ipv4_text_to_int(self, ip_text):
        """convert ip v4 string to integer."""
        if ip_text is None:
            return None
        assert isinstance(ip_text, str)
        return struct.unpack('!I', addrconv.ipv4.text_to_bin(ip_text))[0]