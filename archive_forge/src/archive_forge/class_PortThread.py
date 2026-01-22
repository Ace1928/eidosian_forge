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
class PortThread(object):

    def __init__(self, function):
        super(PortThread, self).__init__()
        self.function = function
        self.thread = None

    def start(self):
        self.stop()
        self.thread = hub.spawn(self.function)

    def stop(self):
        if self.thread is not None:
            hub.kill(self.thread)
            hub.joinall([self.thread])
            self.thread = None