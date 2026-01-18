import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_mpls_tc(self, dp):
    dl_type = 34887
    tc = 3
    headers = [dp.ofproto.OXM_OF_MPLS_TC]
    self._set_verify(headers, tc)
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_mpls_tc(tc)
    self.add_matches(dp, match)