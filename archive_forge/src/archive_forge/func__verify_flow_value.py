import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _verify_flow_value(self, dp, msg):
    stats = msg.body
    verify = self._verify
    if len(verify) != len(stats):
        return 'flow_count is mismatched. verify=%s stats=%s' % (len(verify), len(stats))
    for s in stats:
        v_port = -1
        v = verify.get(s.table_id, None)
        if v:
            v_port = v[3].port
        s_port = s.instructions[0].actions[0].port
        if v_port != s_port:
            return 'port is mismatched. table_id=%s verify=%s, stats=%s' % (s.table_id, v_port, s_port)
    return True