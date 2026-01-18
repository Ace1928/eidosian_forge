import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_dl_src_masked_f0(self, dp):
    dl_src = 'e2:7a:09:79:0b:0f'
    dl_src_bin = self.haddr_to_bin(dl_src)
    mask = 'ff:ff:ff:ff:ff:00'
    mask_bin = self.haddr_to_bin(mask)
    headers = [dp.ofproto.OXM_OF_ETH_SRC, dp.ofproto.OXM_OF_ETH_SRC_W]
    self._set_verify(headers, dl_src_bin, mask_bin, type_='mac')
    match = dp.ofproto_parser.OFPMatch()
    match.set_dl_src_masked(dl_src_bin, mask_bin)
    self.add_matches(dp, match)