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
def transmit_tc_bpdu(self):
    """ Set send_tc_flg to send Topology Change BPDU. """
    if not self.send_tc_flg:
        timer = datetime.timedelta(seconds=self.port_times.max_age + self.port_times.forward_delay)
        self.send_tc_timer = datetime.datetime.today() + timer
        self.send_tc_flg = True