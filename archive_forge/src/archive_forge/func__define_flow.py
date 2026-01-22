import logging
import struct
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller import dpset
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.mac import haddr_to_str
def _define_flow(self, dp):
    in_port = 1
    out_port = 2
    eth_IP = ether.ETH_TYPE_IP
    eth_VLAN = ether.ETH_TYPE_8021Q
    ip_ICMP = inet.IPPROTO_ICMP
    LOG.debug('--- add_flow VLAN(8) to PopVLAN')
    m_vid = 8
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    match.set_dl_type(eth_IP)
    match.set_vlan_vid(m_vid)
    actions = [dp.ofproto_parser.OFPActionPopVlan(), dp.ofproto_parser.OFPActionOutput(out_port, 0)]
    self._add_flow(dp, match, actions)
    LOG.debug('--- add_flow ICMP to PushVLAN(9)')
    s_vid = 9
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    match.set_dl_type(eth_IP)
    match.set_ip_proto(ip_ICMP)
    f = dp.ofproto_parser.OFPMatchField.make(dp.ofproto.OXM_OF_VLAN_VID, s_vid)
    actions = [dp.ofproto_parser.OFPActionPushVlan(eth_VLAN), dp.ofproto_parser.OFPActionSetField(f), dp.ofproto_parser.OFPActionOutput(out_port, 0)]
    self._add_flow(dp, match, actions)
    m_vid = 10
    s_vid = 20
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    match.set_dl_type(eth_IP)
    match.set_vlan_vid(m_vid)
    f = dp.ofproto_parser.OFPMatchField.make(dp.ofproto.OXM_OF_VLAN_VID, s_vid)
    actions = [dp.ofproto_parser.OFPActionPushVlan(eth_VLAN), dp.ofproto_parser.OFPActionSetField(f), dp.ofproto_parser.OFPActionOutput(out_port, 0)]
    LOG.debug('--- add_flow VLAN(100):VLAN to PopVLAN')
    m_vid = 100
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    match.set_dl_type(eth_VLAN)
    match.set_vlan_vid(m_vid)
    actions = [dp.ofproto_parser.OFPActionPopVlan(), dp.ofproto_parser.OFPActionOutput(out_port, 0)]
    self._add_flow(dp, match, actions)