import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_icmpv6_code(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ip_proto = inet.IPPROTO_ICMPV6
    icmp_type = 138
    icmp_code = 1
    headers = [dp.ofproto.OXM_OF_ICMPV6_CODE]
    self._set_verify(headers, icmp_code)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_icmpv6_type(icmp_type)
    match.set_icmpv6_code(icmp_code)
    self.add_matches(dp, match)