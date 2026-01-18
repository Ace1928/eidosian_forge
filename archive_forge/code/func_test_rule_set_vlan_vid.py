import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def test_rule_set_vlan_vid(self, dp):
    vlan_vid = 1263
    headers = [dp.ofproto.OXM_OF_VLAN_VID, dp.ofproto.OXM_OF_VLAN_VID_W]
    self._set_verify(headers, vlan_vid)
    match = dp.ofproto_parser.OFPMatch()
    match.set_vlan_vid(vlan_vid)
    self.add_matches(dp, match)