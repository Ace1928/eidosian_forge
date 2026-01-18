import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_udp_dst(self, dp):
    dl_type = ether.ETH_TYPE_IP
    ip_proto = inet.IPPROTO_UDP
    tp_dst = 61278
    headers = [dp.ofproto.OXM_OF_UDP_DST]
    self._set_verify(headers, tp_dst)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_proto(ip_proto)
    match.set_udp_dst(tp_dst)
    self.add_matches(dp, match)