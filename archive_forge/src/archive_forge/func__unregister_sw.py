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
def _unregister_sw(self, dp):
    if dp.id == self.target_dpid:
        self.target_sw.dp = DummyDatapath()
        msg = 'Leave target SW.'
    elif dp.id == self.tester_dpid:
        self.tester_sw.dp = DummyDatapath()
        msg = 'Leave tester SW.'
    else:
        msg = 'Disconnect unknown SW.'
    if dp.id:
        self.logger.info('dpid=%s : %s', dpid_lib.dpid_to_str(dp.id), msg)