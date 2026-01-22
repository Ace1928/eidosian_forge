import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _add_flow_flow_removed(self, dp, reason, table_id=0, cookie=255, priority=100, in_port=1, idle_timeout=0, hard_timeout=0):
    self._verify = {}
    self._verify['params'] = {'reason': reason, 'table_id': table_id, 'cookie': cookie, 'priority': priority}
    self._verify['in_port'] = in_port
    self._verify['timeout'] = idle_timeout
    if hard_timeout:
        if idle_timeout == 0 or idle_timeout > hard_timeout:
            self._verify['timeout'] = hard_timeout
    match = dp.ofproto_parser.OFPMatch()
    match.set_in_port(in_port)
    self.mod_flow(dp, match=match, cookie=cookie, priority=priority, table_id=table_id, idle_timeout=idle_timeout, hard_timeout=hard_timeout, flags=dp.ofproto.OFPFF_SEND_FLOW_REM)