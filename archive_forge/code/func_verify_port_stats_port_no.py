import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_port_stats_port_no(self, dp, msg):
    ports = msg.body
    if len(ports) > 1:
        return 'reply some ports.\n%s' % ports
    if ports[0].port_no != self._verify:
        return 'port_no is mismatched. request=%s reply=%s' % (self._verify, ports[0].port_no)
    return True