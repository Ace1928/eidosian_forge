import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv6_nd_target(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ip_proto = inet.IPPROTO_ICMPV6
    icmp_type = 135
    target = '5420:db3f:921b:3e33:2791:98f:dd7f:2e19'
    target_int = self.ipv6_to_int(target)
    headers = [dp.ofproto.OXM_OF_IPV6_ND_TARGET]
    self._set_verify(headers, target_int, type_='ipv6')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_icmpv6_type(icmp_type)
    match.set_ipv6_nd_target(target_int)
    self.add_matches(dp, match)