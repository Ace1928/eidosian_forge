import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_port_mod_config_03_mask(self, dp, msg):
    res = self._verify_port_mod_config(dp, msg)
    port_no = self._verify[0]
    p = msg.ports[port_no]
    m = dp.ofproto_parser.OFPPortMod(dp, p.port_no, p.hw_addr, 0, 127, 0)
    dp.send_msg(m)
    dp.send_barrier()
    return res