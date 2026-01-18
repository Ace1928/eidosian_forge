import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_aggregate_stats_flow_count_out_port(self, dp, msg):
    stats = msg.body
    return stats.flow_count == 1