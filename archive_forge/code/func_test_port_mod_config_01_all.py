import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_port_mod_config_01_all(self, dp):
    config = 101
    mask = 127
    self._send_port_mod(dp, config, mask)