import logging
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.tests.integrated import tester
def add_apply_actions(self, dp, actions, match=None):
    inst = [dp.ofproto_parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
    if match is None:
        match = dp.ofproto_parser.OFPMatch()
    m = dp.ofproto_parser.OFPFlowMod(dp, 0, 0, 0, dp.ofproto.OFPFC_ADD, 0, 0, 255, 4294967295, dp.ofproto.OFPP_ANY, dp.ofproto.OFPG_ANY, 0, match, inst)
    dp.send_msg(m)