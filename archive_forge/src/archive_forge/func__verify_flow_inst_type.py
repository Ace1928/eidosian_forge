import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def _verify_flow_inst_type(self, dp, msg):
    inst_type = self._verify
    stats = msg.body
    for s in stats:
        for i in s.instructions:
            if i.type == inst_type:
                return True
    return 'not found inst_type[%s]' % (inst_type,)