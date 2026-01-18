import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def test_port_stats_port_no(self, dp):
    p = self.get_port(dp)
    if not p:
        err = 'need attached port to switch.'
        self.results[self.current] = err
        self.start_next_test(dp)
        return
    self._verify = p.port_no
    m = dp.ofproto_parser.OFPPortStatsRequest(dp, p.port_no)
    dp.send_msg(m)