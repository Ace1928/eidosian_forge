import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _send_packet_out(self, dp, buffer_id=4294967295, in_port=None, output=None, data=''):
    if in_port is None:
        in_port = dp.ofproto.OFPP_LOCAL
    if output is None:
        output = dp.ofproto.OFPP_CONTROLLER
    self._verify['in_port'] = in_port
    self._verify['data'] = data
    actions = [dp.ofproto_parser.OFPActionOutput(output, len(data))]
    m = dp.ofproto_parser.OFPPacketOut(dp, buffer_id, in_port, actions, data)
    dp.send_msg(m)