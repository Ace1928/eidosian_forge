import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_arp_sha(self, dp):
    dl_type = ether.ETH_TYPE_ARP
    arp_sha = '3e:ec:13:9b:f3:0b'
    arp_sha_bin = self.haddr_to_bin(arp_sha)
    headers = [dp.ofproto.OXM_OF_ARP_SHA, dp.ofproto.OXM_OF_ARP_SHA_W]
    self._set_verify(headers, arp_sha_bin, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_arp_sha(arp_sha_bin)
    self.add_matches(dp, match)