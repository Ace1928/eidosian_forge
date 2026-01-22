import logging
from os_ken.tests.integrated import tester
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ether
from os_ken.ofproto import nx_match
def _verify_rule(self, rule, name, value):
    f_value = getattr(rule, name)
    if f_value != value:
        return 'Value error. send:%s=%s val:%s' % (name, value, f_value)
    return True