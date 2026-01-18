import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_vlan_vid_masked_00(self, dp):
    vlan_vid = 1263
    mask = 0
    headers = [dp.ofproto.OXM_OF_VLAN_VID, dp.ofproto.OXM_OF_VLAN_VID_W]
    self._set_verify(headers, vlan_vid, mask)
    match = dp.ofproto_parser.OFPMatch()
    match.set_vlan_vid_masked(vlan_vid, mask)
    self.add_matches(dp, match)