import binascii
import inspect
import json
import logging
import math
import netaddr
import os
import signal
import sys
import time
import traceback
from random import randint
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.lib import dpid as dpid_lib
from os_ken.lib import hub
from os_ken.lib import stringify
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_parser
from os_ken.ofproto import ofproto_protocol
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
from os_ken.ofproto import ofproto_v1_4
from os_ken.ofproto import ofproto_v1_5
def _test_get_throughput(self):
    xid = self.tester_sw.send_flow_stats()
    self.send_msg_xids.append(xid)
    self._wait()
    assert len(self.rcv_msgs) == 1
    flow_stats = self.rcv_msgs[0].body
    self.logger.debug(flow_stats)
    result = {}
    for stat in flow_stats:
        if stat.cookie != THROUGHPUT_COOKIE:
            continue
        result[str(stat.match)] = (stat.byte_count, stat.packet_count)
    return (time.time(), result)