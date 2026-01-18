import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_packet_in_table_id(self, dp):
    in_port = 1
    table_id = 2
    output = dp.ofproto.OFPP_TABLE
    self._verify = {}
    self._verify['reason'] = dp.ofproto.OFPR_ACTION
    self._verify['table_id'] = table_id
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    inst = [dp.ofproto_parser.OFPInstructionGotoTable(table_id)]
    self.mod_flow(dp, inst=inst, match=match)
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    out = dp.ofproto.OFPP_CONTROLLER
    actions = [dp.ofproto_parser.OFPActionOutput(out, 0)]
    self.mod_flow(dp, actions=actions, match=match, table_id=table_id)
    dp.send_barrier()
    self._send_packet_out(dp, in_port=in_port, output=output)