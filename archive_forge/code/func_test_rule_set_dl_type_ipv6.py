import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_rule_set_dl_type_ipv6(self, dp):
    dl_type = ether.ETH_TYPE_IPV6
    self._verify = ['dl_type', dl_type]
    rule = nx_match.ClsRule()
    rule.set_dl_type(dl_type)
    self.add_rule(dp, rule)