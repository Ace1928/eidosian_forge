import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ip_proto_ipv6_frag(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    ip_proto = inet.IPPROTO_FRAGMENT
    headers = [dp.ofproto.OXM_OF_IP_PROTO]
    self._set_verify(headers, ip_proto)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    self.add_matches(dp, match)