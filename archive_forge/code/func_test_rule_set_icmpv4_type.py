import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_icmpv4_type(self, dp):
    dl_type = ether.ETH_TYPE_IP
    ip_proto = inet.IPPROTO_ICMP
    icmp_type = 8
    headers = [dp.ofproto.OXM_OF_ICMPV4_TYPE]
    self._set_verify(headers, icmp_type)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_icmpv4_type(icmp_type)
    self.add_matches(dp, match)