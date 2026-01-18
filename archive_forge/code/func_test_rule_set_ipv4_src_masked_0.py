import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv4_src_masked_0(self, dp):
    dl_type = ether.ETH_TYPE_IP
    src = '192.168.196.250'
    src_int = self.ipv4_to_int(src)
    mask = '0.0.0.0'
    mask_int = self.ipv4_to_int(mask)
    headers = [dp.ofproto.OXM_OF_IPV4_SRC, dp.ofproto.OXM_OF_IPV4_SRC_W]
    self._set_verify(headers, src_int, mask_int, type_='ipv4')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ipv4_src_masked(src_int, mask_int)
    self.add_matches(dp, match)