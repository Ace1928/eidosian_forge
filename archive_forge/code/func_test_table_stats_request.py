import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_table_stats_request(self, dp):
    self._verify = dp.ofproto.OFPST_TABLE
    m = dp.ofproto_parser.OFPTableStatsRequest(dp)
    dp.send_msg(m)