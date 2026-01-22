import logging
import struct
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller import dpset
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ether
from os_ken.ofproto import inet
from os_ken.lib.mac import haddr_to_str
def _add_flow(self, dp, match, actions):
    inst = [dp.ofproto_parser.OFPInstructionActions(dp.ofproto.OFPIT_APPLY_ACTIONS, actions)]
    mod = dp.ofproto_parser.OFPFlowMod(dp, cookie=0, cookie_mask=0, table_id=0, command=dp.ofproto.OFPFC_ADD, idle_timeout=0, hard_timeout=0, priority=255, buffer_id=4294967295, out_port=dp.ofproto.OFPP_ANY, out_group=dp.ofproto.OFPG_ANY, flags=0, match=match, instructions=inst)
    dp.send_msg(mod)