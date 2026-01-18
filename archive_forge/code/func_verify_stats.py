import time
import logging
from os_ken.controller import ofp_event
from os_ken.controller.handler import MAIN_DISPATCHER
from os_ken.controller.handler import set_ev_cls
from os_ken.ofproto import ofproto_v1_2
from os_ken.tests.integrated import tester
def verify_stats(self, dp, stats, type_):
    stats_types = dp.ofproto_parser.OFPStatsReply._STATS_TYPES
    expect = stats_types.get(type_).__name__
    if isinstance(stats, list):
        for s in stats:
            if expect == s.__class__.__name__:
                return True
    elif expect == stats.__class__.__name__:
        return True
    return "Reply msg has not '%s' class.\n%s" % (expect, stats)