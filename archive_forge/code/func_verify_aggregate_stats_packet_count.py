import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_aggregate_stats_packet_count(self, dp, msg):
    for name, val in self._verify.items():
        r_val = getattr(msg.body, name)
        if val != r_val:
            return '%s is mismatched. verify=%s, reply=%s' % (name, val, r_val)
    return True