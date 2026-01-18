import logging
import netaddr
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_2_parser
from os_ken.lib import ofctl_utils
def to_match(dp, attrs):
    convert = {'in_port': UTIL.ofp_port_from_user, 'in_phy_port': str_to_int, 'metadata': ofctl_utils.to_match_masked_int, 'dl_dst': ofctl_utils.to_match_eth, 'dl_src': ofctl_utils.to_match_eth, 'eth_dst': ofctl_utils.to_match_eth, 'eth_src': ofctl_utils.to_match_eth, 'dl_type': str_to_int, 'eth_type': str_to_int, 'dl_vlan': to_match_vid, 'vlan_vid': to_match_vid, 'vlan_pcp': str_to_int, 'ip_dscp': str_to_int, 'ip_ecn': str_to_int, 'nw_proto': str_to_int, 'ip_proto': str_to_int, 'nw_src': ofctl_utils.to_match_ip, 'nw_dst': ofctl_utils.to_match_ip, 'ipv4_src': ofctl_utils.to_match_ip, 'ipv4_dst': ofctl_utils.to_match_ip, 'tp_src': str_to_int, 'tp_dst': str_to_int, 'tcp_src': str_to_int, 'tcp_dst': str_to_int, 'udp_src': str_to_int, 'udp_dst': str_to_int, 'sctp_src': str_to_int, 'sctp_dst': str_to_int, 'icmpv4_type': str_to_int, 'icmpv4_code': str_to_int, 'arp_op': str_to_int, 'arp_spa': ofctl_utils.to_match_ip, 'arp_tpa': ofctl_utils.to_match_ip, 'arp_sha': ofctl_utils.to_match_eth, 'arp_tha': ofctl_utils.to_match_eth, 'ipv6_src': ofctl_utils.to_match_ip, 'ipv6_dst': ofctl_utils.to_match_ip, 'ipv6_flabel': str_to_int, 'icmpv6_type': str_to_int, 'icmpv6_code': str_to_int, 'ipv6_nd_target': ofctl_utils.to_match_ip, 'ipv6_nd_sll': ofctl_utils.to_match_eth, 'ipv6_nd_tll': ofctl_utils.to_match_eth, 'mpls_label': str_to_int, 'mpls_tc': str_to_int}
    keys = {'dl_dst': 'eth_dst', 'dl_src': 'eth_src', 'dl_type': 'eth_type', 'dl_vlan': 'vlan_vid', 'nw_src': 'ipv4_src', 'nw_dst': 'ipv4_dst', 'nw_proto': 'ip_proto'}
    if attrs.get('dl_type') == ether.ETH_TYPE_ARP or attrs.get('eth_type') == ether.ETH_TYPE_ARP:
        if 'nw_src' in attrs and 'arp_spa' not in attrs:
            attrs['arp_spa'] = attrs['nw_src']
            del attrs['nw_src']
        if 'nw_dst' in attrs and 'arp_tpa' not in attrs:
            attrs['arp_tpa'] = attrs['nw_dst']
            del attrs['nw_dst']
    kwargs = {}
    for key, value in attrs.items():
        if key in keys:
            key = keys[key]
        if key in convert:
            value = convert[key](value)
            if key == 'tp_src' or key == 'tp_dst':
                conv = {inet.IPPROTO_TCP: {'tp_src': 'tcp_src', 'tp_dst': 'tcp_dst'}, inet.IPPROTO_UDP: {'tp_src': 'udp_src', 'tp_dst': 'udp_dst'}}
                ip_proto = attrs.get('nw_proto', attrs.get('ip_proto', 0))
                key = conv[ip_proto][key]
                kwargs[key] = value
            else:
                kwargs[key] = value
        else:
            LOG.error('Unknown match field: %s', key)
    return dp.ofproto_parser.OFPMatch(**kwargs)