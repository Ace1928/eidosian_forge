import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def test_rule_set_in_port(self, dp):
    in_port = 32
    self._verify = ['in_port', in_port]
    rule = nx_match.ClsRule()
    rule.set_in_port(in_port)
    self.add_rule(dp, rule)