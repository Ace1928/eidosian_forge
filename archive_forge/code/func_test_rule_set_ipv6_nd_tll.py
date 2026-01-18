import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_nd_tll(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ip_proto = inet.IPPROTO_ICMPV6
    icmp_type = 136
    nd_tll = '18:f6:66:b6:f1:b3'
    nd_tll_bin = self.haddr_to_bin(nd_tll)
    headers = [dp.ofproto.OXM_OF_IPV6_ND_TLL]
    self._set_verify(headers, nd_tll_bin, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_icmpv6_type(icmp_type)
    match.set_ipv6_nd_tll(nd_tll_bin)
    self.add_matches(dp, match)