import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_arp_tha_masked_00(self, dp):
    dl_type = ether.ETH_TYPE_ARP
    arp_tha = '83:6c:21:52:49:68'
    arp_tha_bin = self.haddr_to_bin(arp_tha)
    mask = '00:00:00:00:00:00'
    mask_bin = self.haddr_to_bin(mask)
    headers = [dp.ofproto.OXM_OF_ARP_THA, dp.ofproto.OXM_OF_ARP_THA_W]
    self._set_verify(headers, arp_tha_bin, mask_bin, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_arp_tha_masked(arp_tha_bin, mask_bin)
    self.add_matches(dp, match)