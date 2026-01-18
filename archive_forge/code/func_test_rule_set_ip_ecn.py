import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ip_ecn(self, dp):
    dl_type = ether.ETH_TYPE_IP
    ip_ecn = 3
    headers = [dp.ofproto.OXM_OF_IP_ECN]
    self._set_verify(headers, ip_ecn)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ip_ecn(ip_ecn)
    self.add_matches(dp, match)