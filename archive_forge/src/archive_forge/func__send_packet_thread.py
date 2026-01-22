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
def _send_packet_thread(self, arg):
    """ Send several packets continuously. """
    if self.ingress_event is None or self.ingress_event._cond:
        return
    if not arg['thread_counter'] % arg['dot_span']:
        sys.stdout.write('.')
        sys.stdout.flush()
    arg['thread_counter'] += 1
    arg['packet_counter'] += arg['packet_counter_inc']
    count = int(arg['packet_counter'])
    arg['packet_counter'] -= count
    hub.sleep(CONTINUOUS_THREAD_INTVL)
    tid = hub.spawn(self._send_packet_thread, arg)
    self.ingress_threads.append(tid)
    hub.sleep(0)
    for _ in range(count):
        if arg['randomize']:
            msg = eval('/'.join(arg['packet_text']))
            msg.serialize()
            data = msg.data
        else:
            data = arg['packet_binary']
        try:
            self.tester_sw.send_packet_out(data)
        except Exception as err:
            self.thread_msg = err
            self.ingress_event.set()
            break