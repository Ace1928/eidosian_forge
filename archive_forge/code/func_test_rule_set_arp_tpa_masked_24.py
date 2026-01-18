import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_arp_tpa_masked_24(self, dp):
    dl_type = ether.ETH_TYPE_ARP
    nw_dst = '192.168.198.233'
    nw_dst_int = self.ipv4_to_int(nw_dst)
    mask = '255.255.255.0'
    mask_int = self.ipv4_to_int(mask)
    headers = [dp.ofproto.OXM_OF_ARP_TPA, dp.ofproto.OXM_OF_ARP_TPA_W]
    self._set_verify(headers, nw_dst_int, mask_int, type_='ipv4')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_type(dl_type)
    match.set_arp_tpa_masked(nw_dst_int, mask_int)
    self.add_matches(dp, match)