import datetime
import logging
from os_ken.base import app_manager
from os_ken.controller import event
from os_ken.controller import handler
from os_ken.controller import ofp_event
from os_ken.controller.handler import set_ev_cls
from os_ken.exception import OSKenException
from os_ken.exception import OFPUnknownVersion
from os_ken.lib import hub
from os_ken.lib import mac
from os_ken.lib.dpid import dpid_to_str
from os_ken.lib.packet import bpdu
from os_ken.lib.packet import ethernet
from os_ken.lib.packet import llc
from os_ken.lib.packet import packet
from os_ken.ofproto import ofproto_v1_0
from os_ken.ofproto import ofproto_v1_2
from os_ken.ofproto import ofproto_v1_3
def _wait_bpdu_timer(self):
    time_exceed = False
    while True:
        self.wait_timer_event = hub.Event()
        message_age = self.designated_times.message_age if self.designated_times else 0
        timer = self.port_times.max_age - message_age
        timeout = hub.Timeout(timer)
        try:
            self.wait_timer_event.wait()
        except hub.Timeout as t:
            if t is not timeout:
                err_msg = 'Internal error. Not my timeout.'
                raise OSKenException(msg=err_msg)
            self.logger.info('[port=%d] Wait BPDU timer is exceeded.', self.ofport.port_no, extra=self.dpid_str)
            time_exceed = True
        finally:
            timeout.cancel()
            self.wait_timer_event = None
        if time_exceed:
            break
    if time_exceed:
        hub.spawn(self.wait_bpdu_timeout)