import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def add_set_field_action(self, dp, field, value, match=None):
    self._verify = [dp.ofproto.OFPAT_SET_FIELD, 'field', field, value]
    f = dp.ofproto_parser.OFPMatchField.make(field, value)
    actions = [dp.ofproto_parser.OFPActionSetField(f)]
    self.add_apply_actions(dp, actions, match=match)