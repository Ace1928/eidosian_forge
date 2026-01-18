import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_packet_in_action(self, dp):
    self._verify = {}
    self._verify['reason'] = dp.ofproto.OFPR_ACTION
    self._send_packet_out(dp)