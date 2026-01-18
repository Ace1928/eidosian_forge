import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_nd_sll(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ip_proto = inet.IPPROTO_ICMPV6
    icmp_type = 135
    nd_sll = '93:6d:d0:d4:e8:36'
    nd_sll_bin = self.haddr_to_bin(nd_sll)
    headers = [dp.ofproto.OXM_OF_IPV6_ND_SLL]
    self._set_verify(headers, nd_sll_bin, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_icmpv6_type(icmp_type)
    match.set_ipv6_nd_sll(nd_sll_bin)
    self.add_matches(dp, match)