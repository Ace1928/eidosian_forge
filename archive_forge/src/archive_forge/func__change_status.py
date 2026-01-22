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
def _change_status(self, new_state, thread_switch=True):
    if new_state is not PORT_STATE_DISABLE:
        self.ofctl.set_port_status(self.ofport, new_state)
    if new_state is PORT_STATE_FORWARD or (self.state is PORT_STATE_FORWARD and (new_state is PORT_STATE_DISABLE or new_state is PORT_STATE_BLOCK)):
        self.topology_change_notify(new_state)
    if new_state is PORT_STATE_DISABLE or new_state is PORT_STATE_BLOCK:
        self.send_tc_flg = False
        self.send_tc_timer = None
        self.send_tcn_flg = False
        self.send_bpdu_thread.stop()
    elif new_state is PORT_STATE_LISTEN:
        self.send_bpdu_thread.start()
    self.state = new_state
    self.send_event(EventPortStateChange(self.dp, self))
    if self.state_event is not None:
        self.state_event.set()
        self.state_event = None
    if thread_switch:
        hub.sleep(0)