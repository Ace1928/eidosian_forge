import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_ipv4_dst(self, dp):
    dl_type = ether.ETH_TYPE_IP
    dst = '192.168.54.155'
    dst_int = self.ipv4_to_int(dst)
    headers = [dp.ofproto.OXM_OF_IPV4_DST, dp.ofproto.OXM_OF_IPV4_DST_W]
    self._set_verify(headers, dst_int, type_='ipv4')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_ipv4_dst(dst_int)
    self.add_matches(dp, match)